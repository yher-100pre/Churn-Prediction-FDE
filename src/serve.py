"""FastAPI churn prediction service.

Serves the artifacts frozen by train.py: the XGBoost booster, the operating
threshold, and the feature contract. Three endpoints -- /health, /score/single,
/score/batch -- behind JWT auth, a per-IP rate limit, and structured JSON logs.

Three details from the training contract drive most of the code below, and each
one is a silent-wrongness bug if it is dropped:

1. COLUMN ORDER IS CONTRACTUAL. feature_spec.json publishes feature_cols and
   says so explicitly. XGBoost indexes positionally, so a matrix assembled in
   dict-iteration order scores confidently and wrongly. Every matrix here is
   built by reindexing on FEATURE_COLS, never by trusting request key order.

2. THE MODEL IS EARLY-STOPPED AT best_iteration=71 (of 92 saved rounds).
   save_model preserves that attribute and predict_proba honours it, so the
   loaded model reproduces training scores exactly -- which is the only reason
   the frozen threshold (0.786) still means what threshold.json says it means.
   Never pass an explicit iteration_range here; it would silently score with all
   92 trees and invalidate the threshold. shap.TreeExplainer honours
   best_iteration too, so the explanation and the probability agree.

3. BATCH USES TOP-K RANK SELECTION, NOT THE THRESHOLD. 72 shallow trees produce
   clustered scores, so a `>= threshold` cut overshoots the campaign budget by
   5-16% (threshold.json). Batch replicates train.py's selection exactly:
   k = int(round(rate * n)) over np.argsort(-scores, kind="stable"). The
   threshold remains the fallback for single scoring, where there is no
   population to rank against.

Run locally:
    export CHURN_SERVICE_SECRET="dev-secret-do-not-use-in-prod"
    python src/serve.py                # serves on :8000
    python src/serve.py --make-token   # prints a signed test token, then exits
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import uvicorn
import xgboost as xgb
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field, field_validator

# --- Configuration -----------------------------------------------------------

MODELS_DIR = Path(os.environ.get("CHURN_MODELS_DIR", "models"))

ARTIFACTS = {
    "model": MODELS_DIR / "churn_model.json",
    "baseline_lr": MODELS_DIR / "baseline_lr.joblib",
    "feature_spec": MODELS_DIR / "feature_spec.json",
    "threshold": MODELS_DIR / "threshold.json",
    "model_card": MODELS_DIR / "model_card.json",
}

JWT_ALGORITHM = "HS256"

RATE_LIMIT_MAX_REQUESTS = 100
RATE_LIMIT_WINDOW_SEC = 60

MAX_BATCH_SIZE = 1000

# The linear baseline was fitted on TEN columns: the nine contractual features
# plus an explicit no-prior-session indicator, with recency capped. XGBoost reads
# the 999.0 sentinel as its own regime; a linear model would read it as a literal
# 999-day distance and let it dominate the coefficient (see train.py's
# LR_NO_SESSION_COL). Serving the baseline means reproducing that transform --
# feeding it the raw nine-column matrix would score silently and wrongly.
LR_NO_SESSION_COL = "has_no_prior_session"
LR_RECENCY_CAP = 365.0

# Trusting X-Forwarded-For lets any client spoof its IP and walk around the rate
# limit, so it is off unless the service is genuinely behind a proxy that
# overwrites the header (an ALB or ingress does).
TRUST_PROXY_HEADERS = os.environ.get("CHURN_TRUST_PROXY_HEADERS", "").lower() in {"1", "true", "yes"}


def _require_secret() -> str:
    """Read the signing secret from the environment, or refuse to start.

    Defaulting to a baked-in string would mean a misconfigured deploy still
    boots and still validates tokens -- signed with a secret that is public in
    the repo. Failing here is the only way the mistake is visible.
    """
    secret = os.environ.get("CHURN_SERVICE_SECRET")
    if not secret:
        raise RuntimeError(
            "CHURN_SERVICE_SECRET is not set -- refusing to start.\n"
            "The JWT signing secret has no safe default: a hardcoded fallback would "
            "accept tokens anyone with the source could mint.\n"
            'Local testing:  export CHURN_SERVICE_SECRET="dev-secret-do-not-use-in-prod"'
        )
    return secret


# --- Structured logging ------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Render each record as a single JSON object.

    One line per event, so CloudWatch/Athena can parse the stream without a
    grok pattern. Anything passed via `extra` is merged into the object.
    """

    RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            # The traceback goes to the log, never to the client.
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # uvicorn ships its own human-readable handlers; drop them so the stream
    # stays machine-parseable rather than half JSON and half plain text.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(noisy)
        lg.handlers = []
        lg.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    return logging.getLogger("churn.serve")


log = _configure_logging()


# --- Loaded state ------------------------------------------------------------


class ServiceState:
    """Artifacts loaded once at startup and read-only thereafter."""

    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None
        self.baseline_lr: Any = None
        self.explainer: shap.TreeExplainer | None = None
        self.feature_cols: list[str] = []
        self.feature_spec: dict[str, Any] = {}
        self.threshold: float = 0.0
        self.threshold_spec: dict[str, Any] = {}
        self.model_card: dict[str, Any] = {}
        self.target_selection_rate: float = 0.30
        self.model_version: str = "unknown"
        self.started_monotonic: float = time.monotonic()
        self.started_utc: str = ""


STATE = ServiceState()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def load_artifacts(state: ServiceState) -> None:
    """Load every artifact, or raise. Called from the lifespan hook.

    A service that starts with a missing model and only discovers it on the
    first request has turned a deploy-time failure into a production incident,
    so every artifact is checked up front and any miss is fatal.
    """
    missing = [str(p) for p in ARTIFACTS.values() if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing model artifacts -- refusing to start:\n  "
            + "\n  ".join(missing)
            + "\n\nTrain first:  python src/train.py"
            + f"\n(looked in {MODELS_DIR.resolve()}; override with CHURN_MODELS_DIR)"
        )

    state.feature_spec = _load_json(ARTIFACTS["feature_spec"])
    state.feature_cols = list(state.feature_spec["feature_cols"])
    if not state.feature_cols:
        raise RuntimeError(f"{ARTIFACTS['feature_spec']} lists no feature_cols.")

    state.threshold_spec = _load_json(ARTIFACTS["threshold"])
    state.threshold = float(state.threshold_spec["threshold"])
    state.target_selection_rate = float(state.threshold_spec.get("target_selection_rate", 0.30))

    state.model_card = _load_json(ARTIFACTS["model_card"])
    # model_card.json has no explicit version field; the git SHA it was trained
    # at is the closest thing to one and is what makes a score reproducible.
    state.model_version = str(state.model_card.get("git_sha", "unknown"))

    model = xgb.XGBClassifier()
    model.load_model(str(ARTIFACTS["model"]))
    state.model = model

    # Cross-check the booster against the published contract. These two can only
    # disagree if the artifacts were rebuilt out of step with each other, and the
    # failure mode is silent mis-scoring, so it is worth one comparison here.
    booster_names = model.get_booster().feature_names
    if booster_names is not None and list(booster_names) != state.feature_cols:
        raise RuntimeError(
            "Feature mismatch between churn_model.json and feature_spec.json -- refusing to start.\n"
            f"  booster:      {list(booster_names)}\n"
            f"  feature_spec: {state.feature_cols}\n"
            "These artifacts came from different training runs."
        )

    state.baseline_lr = joblib.load(ARTIFACTS["baseline_lr"])

    # The request schema is hand-declared; this is what keeps it honest.
    _validate_schema_against_spec(state)

    # Built once: TreeExplainer's setup walks the whole forest, and paying that
    # per request would dominate single-scoring latency.
    state.explainer = shap.TreeExplainer(model)

    state.started_monotonic = time.monotonic()
    state.started_utc = datetime.now(timezone.utc).isoformat()

    log.info(
        "artifacts_loaded",
        extra={
            "event": "startup",
            "model_version": state.model_version,
            "n_features": len(state.feature_cols),
            "threshold": state.threshold,
            "best_iteration": int(getattr(model, "best_iteration", -1)),
            "target_selection_rate": state.target_selection_rate,
            "models_dir": str(MODELS_DIR.resolve()),
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _require_secret()  # fail before binding the port, not on the first request
    load_artifacts(STATE)
    yield
    log.info("shutdown", extra={"event": "shutdown"})


app = FastAPI(
    title="Churn Prediction Service",
    description="Scores mobile-marketing customers for churn risk from a 9-feature RFM vector.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Auth --------------------------------------------------------------------

# auto_error=False so a missing header lands in our own structured 401 rather
# than FastAPI's default {"detail": ...} shape.
bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    """Validate the bearer token. An `exp` claim is mandatory."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token. Send: Authorization: Bearer <jwt>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = jwt.decode(
            credentials.credentials,
            _require_secret(),
            algorithms=[JWT_ALGORITHM],
            # require_exp: a token without an expiry never stops being valid, so
            # one leaked token is a permanent credential. verify_aud off because
            # this service mints its own tokens and sets no audience.
            options={"require_exp": True, "verify_aud": False},
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return claims


def make_token(subject: str = "local-test", ttl_minutes: int = 60) -> str:
    """Mint a short-lived token. Test helper -- there is no user store here."""
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": subject, "iat": now, "exp": now + timedelta(minutes=ttl_minutes)},
        _require_secret(),
        algorithm=JWT_ALGORITHM,
    )


# --- Rate limiting -----------------------------------------------------------


class SlidingWindowRateLimiter:
    """In-memory sliding-window limiter keyed by client IP.

    Scope, stated plainly because it is easy to over-trust: the counter lives in
    this process. Two uvicorn workers or two pods mean each replica allows the
    full quota independently, so the effective global limit is
    limit x replicas. That is fine as a per-instance abuse brake and is NOT a
    quota mechanism -- a real one needs shared state (Redis/ElastiCache) or an
    API-gateway usage plan.
    """

    def __init__(self, max_requests: int, window_sec: int) -> None:
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._hits: dict[str, deque[float]] = {}
        # Endpoints run in a threadpool, so this dict is genuinely shared state.
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def check(self, key: str, now: float) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        cutoff = now - self.window_sec
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                # The oldest hit in the window is what has to age out.
                retry_after = max(1, int(bucket[0] + self.window_sec - now) + 1)
                return False, retry_after
            bucket.append(now)
            return True, 0

    def maybe_sweep(self, now: float) -> None:
        """Drop buckets with no hits in the window, so idle IPs are not a leak.

        Without this the dict grows one entry per distinct IP forever, which is
        a slow memory leak in a long-lived pod. Runs at most once per window.
        """
        with self._lock:
            if now - self._last_sweep < self.window_sec:
                return
            self._last_sweep = now
            cutoff = now - self.window_sec
            for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
                self._hits.pop(key, None)


rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SEC)


def client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Left-most entry is the original client; the rest are proxy hops.
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --- Schemas -----------------------------------------------------------------


class Features(BaseModel):
    """The nine contractual RFM features.

    Declared explicitly rather than as a free dict[str, float] so that a typo'd
    or missing feature is a 422 with the offending name, not a silent NaN that
    XGBoost happily routes down its missing-value branch and scores anyway.
    Field names and order mirror feature_spec.json; that agreement is asserted
    at startup by _validate_schema_against_spec.
    """

    model_config = {"extra": "forbid"}

    recency_days: float = Field(..., description="Days since last session before the anchor; 999.0 = no prior session")
    freq_30d: float = Field(..., ge=0)
    freq_90d: float = Field(..., ge=0)
    monetary_90d: float = Field(..., ge=0)
    purchase_count_90d: float = Field(..., ge=0)
    avg_session_duration: float = Field(..., ge=0)
    push_open_rate: float = Field(..., ge=0, le=1)
    campaign_click_rate: float = Field(..., ge=0, le=1)
    support_ticket_count: float = Field(..., ge=0)

    @field_validator("*")
    @classmethod
    def _finite(cls, v: float) -> float:
        # NaN/inf would propagate into the score as a missing-value split rather
        # than an error, so it is rejected at the edge.
        if not np.isfinite(v):
            raise ValueError("must be a finite number")
        return v


class SingleScoreRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=128)
    features: Features


class ShapDriver(BaseModel):
    feature: str
    direction: str = Field(..., description="'increases churn risk' or 'decreases churn risk'")
    magnitude: float = Field(..., description="Absolute SHAP contribution, in log-odds")


class SingleScoreResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_predicted: bool
    threshold_used: float
    top_shap_drivers: list[ShapDriver]
    recommendation: str


class BatchCustomer(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=128)
    features: Features


class BatchScoreRequest(BaseModel):
    customers: list[BatchCustomer] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class ScoredCustomer(BaseModel):
    customer_id: str
    churn_probability: float
    rank: int = Field(..., description="1 = highest risk")
    selected: bool = Field(..., description="Inside the top-K retention budget")


class BatchScoreResponse(BaseModel):
    scored: list[ScoredCustomer]
    total: int
    contacted_at_30pct: int
    batch_selection_rate: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: str


def _validate_schema_against_spec(state: ServiceState) -> None:
    """Assert the Pydantic model matches feature_spec.json exactly.

    The request schema is hand-declared while the matrix order comes from the
    spec. If a retrain adds or renames a feature, this is what turns that into a
    startup failure instead of a 422 storm (or worse, a quietly wrong column).
    """
    declared = list(Features.model_fields.keys())
    if declared != state.feature_cols:
        raise RuntimeError(
            "Features schema does not match feature_spec.json -- refusing to start.\n"
            f"  schema:       {declared}\n"
            f"  feature_spec: {state.feature_cols}\n"
            "Update the Features model in serve.py to match the retrained contract."
        )


# --- Scoring -----------------------------------------------------------------


def build_matrix(rows: list[Features], state: ServiceState) -> pd.DataFrame:
    """Assemble the feature matrix in the contractual column order.

    Reindexing on feature_cols is the whole point: request key order is not the
    training order, and XGBoost would not complain about the difference.
    """
    frame = pd.DataFrame([r.model_dump() for r in rows])
    return frame.reindex(columns=state.feature_cols).astype(float)


def to_lr_matrix(matrix: pd.DataFrame, state: ServiceState) -> pd.DataFrame:
    """Apply the linear baseline's preprocessing (see LR_NO_SESSION_COL note)."""
    sentinel = float(state.feature_spec.get("no_session_recency_sentinel", 999.0))
    lr = matrix.copy()
    lr[LR_NO_SESSION_COL] = (lr["recency_days"] == sentinel).astype(float)
    lr["recency_days"] = lr["recency_days"].clip(upper=LR_RECENCY_CAP)
    return lr


def predict_proba(matrix: pd.DataFrame, state: ServiceState) -> np.ndarray:
    """Churn probabilities. No iteration_range -- see module docstring, point 2."""
    return state.model.predict_proba(matrix)[:, 1]


def top_shap_drivers(matrix: pd.DataFrame, state: ServiceState, top_n: int = 3) -> list[ShapDriver]:
    """Per-request TreeExplainer attribution for a single row.

    Exact for tree ensembles, and signed in log-odds of the positive (churn)
    class, so a positive contribution is unambiguously risk-increasing.
    """
    values = state.explainer.shap_values(matrix)
    values = np.asarray(values)
    if values.ndim == 3:
        # Some SHAP/XGBoost combinations return (n, features, classes).
        values = values[..., -1]
    row = values[0]

    order = np.argsort(-np.abs(row))[:top_n]
    return [
        ShapDriver(
            feature=state.feature_cols[i],
            direction="increases churn risk" if row[i] > 0 else "decreases churn risk",
            magnitude=round(float(abs(row[i])), 6),
        )
        for i in order
    ]


def select_top_k(scores: np.ndarray, rate: float) -> tuple[np.ndarray, int]:
    """Rank selection, replicating train.py's _select_top_k exactly.

    k = int(round(rate * n)) over a STABLE argsort of negated scores. Stability
    matters: 72 shallow trees produce clustered and frequently tied scores, and
    an unstable sort would reshuffle tied customers between otherwise identical
    requests. Ties at the boundary are broken by input position -- arbitrary,
    but deterministic and identical to how the model was evaluated.

    Returns (order, k) where order is row indices, highest risk first.
    """
    n = len(scores)
    k = int(round(rate * n))
    k = max(0, min(k, n))
    order = np.argsort(-scores, kind="stable")
    return order, k


# --- Middleware: rate limit + one structured log line per request -------------


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    start = time.perf_counter()
    now = time.time()
    ip = client_ip(request)
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.log_extra = {}

    # Health is exempt: liveness/readiness probes come from a single node IP and
    # would otherwise burn the quota and get themselves throttled, which reads to
    # the orchestrator as an unhealthy pod and triggers restarts.
    if request.url.path != "/health":
        allowed, retry_after = rate_limiter.check(ip, now)
        if not allowed:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log.warning(
                "rate_limit_exceeded",
                extra={
                    "event": "request",
                    "request_id": request_id,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "client_ip": ip,
                    "latency_ms": latency_ms,
                    "status_code": 429,
                    "retry_after": retry_after,
                },
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": (
                        f"Limit is {RATE_LIMIT_MAX_REQUESTS} requests per "
                        f"{RATE_LIMIT_WINDOW_SEC}s per client IP. Retry in {retry_after}s."
                    ),
                },
                headers={"Retry-After": str(retry_after), "X-Request-ID": request_id},
            )

    try:
        response = await call_next(request)
    except Exception:
        # The exception handlers below normally catch this; if the failure
        # happened in middleware, log it here so no request goes unrecorded.
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        log.exception(
            "request_failed",
            extra={
                "event": "request",
                "request_id": request_id,
                "endpoint": request.url.path,
                "method": request.method,
                "client_ip": ip,
                "latency_ms": latency_ms,
                "status_code": 500,
            },
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "Unhandled server error."},
            headers={"X-Request-ID": request_id},
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    record = {
        "event": "request",
        "request_id": request_id,
        "endpoint": request.url.path,
        "method": request.method,
        "client_ip": ip,
        "latency_ms": latency_ms,
        "status_code": response.status_code,
    }
    # Endpoint-specific fields (customer_id / batch_size / scores) are stashed on
    # request.state by the handler so they land in the same line as the latency.
    record.update(getattr(request.state, "log_extra", {}) or {})
    log.info("request", extra=record)

    response.headers["X-Request-ID"] = request_id

    rate_limiter.maybe_sweep(now)

    return response


# --- Error handling ----------------------------------------------------------
# Every error leaves as {"error": ..., "detail": ...}. Tracebacks go to the log.


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    codes = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "payload_too_large",
        422: "validation_error",
        429: "rate_limit_exceeded",
        503: "service_unavailable",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": codes.get(exc.status_code, "error"), "detail": str(exc.detail)},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    problems = [
        f"{'.'.join(str(p) for p in err.get('loc', ()) if p != 'body')}: {err.get('msg')}"
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "detail": "; ".join(problems) or "Request body failed validation.",
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception(
        "unhandled_exception",
        extra={
            "event": "error",
            "request_id": getattr(request.state, "request_id", None),
            "endpoint": request.url.path,
            "exception_type": type(exc).__name__,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "An unexpected error occurred. The incident was logged server-side.",
        },
    )


# --- Endpoints ---------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness. No auth -- probes cannot carry a token."""
    return HealthResponse(
        status="ok" if STATE.model is not None else "degraded",
        model_version=STATE.model_version,
        uptime_seconds=round(time.monotonic() - STATE.started_monotonic, 3),
    )


@app.post(
    "/score/single",
    response_model=SingleScoreResponse,
    tags=["scoring"],
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
def score_single(
    payload: SingleScoreRequest,
    request: Request,
    claims: dict = Depends(require_auth),
) -> SingleScoreResponse:
    """Score one customer, with SHAP attribution.

    Defined with `def`, not `async def`: predict_proba and TreeExplainer are
    blocking CPU work, and on the event loop they would stall every concurrent
    request. FastAPI runs sync handlers in a threadpool instead.

    This is the one path that uses the frozen threshold rather than rank
    selection -- there is no population to rank a single customer against.
    """
    matrix = build_matrix([payload.features], STATE)
    probability = float(predict_proba(matrix, STATE)[0])
    drivers = top_shap_drivers(matrix, STATE)

    predicted = probability >= STATE.threshold
    recommendation = (
        "high_risk — prioritize for retention campaign"
        if predicted
        else "low_risk — monitor"
    )

    request.state.log_extra = {
        "customer_id": payload.customer_id,
        "churn_probability": round(probability, 6),
        "churn_predicted": predicted,
        "subject": claims.get("sub"),
    }

    return SingleScoreResponse(
        customer_id=payload.customer_id,
        churn_probability=round(probability, 6),
        churn_predicted=predicted,
        threshold_used=STATE.threshold,
        top_shap_drivers=drivers,
        recommendation=recommendation,
    )


@app.post(
    "/score/batch",
    response_model=BatchScoreResponse,
    tags=["scoring"],
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
def score_batch(
    payload: BatchScoreRequest,
    request: Request,
    claims: dict = Depends(require_auth),
) -> BatchScoreResponse:
    """Score a population and select the top-K by rank, not by threshold.

    Rank selection is what makes the 30% campaign budget exact. A `>= threshold`
    cut on this model overshoots by 5-16% because the scores cluster
    (threshold.json), and an overshoot is real money in a retention campaign.

    Note the budget rounds: k = round(0.30 * n), so a 3-customer batch contacts
    1 and a 1-customer batch contacts 0. That is the documented policy, and
    /score/single is the endpoint for a lone customer.
    """
    customers = payload.customers
    matrix = build_matrix([c.features for c in customers], STATE)

    # One vectorised call for the whole batch: 1000 individual predict_proba
    # calls would be dominated by per-call overhead.
    scores = predict_proba(matrix, STATE)

    order, k = select_top_k(scores, STATE.target_selection_rate)
    total = len(customers)

    scored = [
        ScoredCustomer(
            customer_id=customers[idx].customer_id,
            churn_probability=round(float(scores[idx]), 6),
            rank=position + 1,
            selected=position < k,
        )
        for position, idx in enumerate(order)
    ]

    request.state.log_extra = {
        "batch_size": total,
        "top_score": round(float(scores[order[0]]), 6) if total else None,
        "contacted_at_30pct": k,
        "subject": claims.get("sub"),
    }

    return BatchScoreResponse(
        scored=scored,
        total=total,
        contacted_at_30pct=k,
        batch_selection_rate=round(k / total, 6) if total else 0.0,
    )


# --- Entrypoint --------------------------------------------------------------

if __name__ == "__main__":
    if "--make-token" in sys.argv:
        # Convenience for local testing; the service itself never mints tokens.
        print(make_token())
        sys.exit(0)

    _require_secret()  # surface the misconfiguration before uvicorn binds
    # The app object, not an import string: an import string would re-import this
    # file as a second module and build a second, unrelated STATE.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # Single worker on purpose: the rate limiter is per-process, so extra
        # workers would multiply the effective limit (see SlidingWindowRateLimiter).
        workers=1,
        log_config=None,  # keep our JSON formatter, not uvicorn's
    )
