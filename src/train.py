"""Model training: RFM feature table -> churn model, baselines, and audit artifacts.

Trains an XGBoost churn classifier against two baselines (a recency rule and a
logistic regression), selects an operating threshold from a campaign budget
constraint, explains the model with SHAP, audits false-negative rates across
synthetic segments, and writes everything the scoring service needs to load and
run the model in production.

Reads data/training_features.parquet (produced by generate_synthetic.main, which
in turn calls feature_engineering.build_features). Writes to models/, which is
gitignored -- artifacts belong in S3, not the repo.

The churn class is the MAJORITY class here
------------------------------------------
This dataset runs about 64% churned / 36% retained, which inverts the usual
churn-modelling reflexes and is the single easiest thing to get wrong:

- Accuracy has a floor of ~0.64 from the constant "everyone churns" rule. An
  accuracy of 0.72 sounds respectable and is nearly worthless. Every accuracy
  figure this module reports is printed next to that floor, and accuracy is
  never the headline metric.
- scale_pos_weight is pinned at 1.0. The reflex value (neg/pos) would be ~0.56,
  and the reflex *direction* -- upweighting the positive class -- is simply wrong
  when that class already dominates. Either would distort the predicted
  probabilities, and this model's probabilities are load-bearing: the threshold
  policy below reads them as a ranking, and calibration is reported as a metric.
- PR-AUC on the churn class has a no-skill baseline of ~0.64, so it looks
  flattering and says little. PR-AUC on the RETENTION class is the informative
  one, because "who will stay" is the genuinely rare and harder call.

Ranking quality, not label accuracy, is what the downstream use case consumes.

Threshold policy: top-K budget constraint
-----------------------------------------
The score feeds campaign audience selection, and the campaign can contact 30% of
the base per cycle (TARGET_SELECTION_RATE). So the operating threshold is not
0.5 and is not tuned for F1: it is the quantile of predicted probability that
selects exactly the top 30% of customers by churn risk.

That makes the threshold a business input rather than a hyperparameter, and it
means the model only has to RANK well within the region that matters -- absolute
probability calibration affects reporting, not audience membership. The chosen
threshold and the policy that produced it are written to threshold.json so the
service applies the same cut-off it was evaluated at. A service that re-derives
its own threshold is a service running an unevaluated model.

Train/serve skew: recency is anchored differently in each
---------------------------------------------------------
feature_engineering anchors every feature at CUTOFF for training, deliberately,
so post-cutoff activity cannot leak into features that predict a post-cutoff
label. Production scoring has no such constraint and anchors recency at the
current date. Both are correct for their context, but they are NOT the same
distribution: a production customer's recency_days is measured from today, and
the gap between today and CUTOFF grows every day the model stays deployed.

This is a real and expected drift, not a bug to fix here. What this module owes
the service is the evidence to detect it: feature_spec.json records CUTOFF, what
it anchors, and the training recency distribution (mean / p50 / p90), so a
monitor can compare live feature distributions against the ones the model was
fitted on and alarm when they diverge.

Segments are audit dimensions, never features
---------------------------------------------
plan_tier, acquisition_channel and region are EXCLUDED from FEATURE_COLS. They
are invented (see the generate_synthetic docstring), and their association with
churn was written into the generator by hand. A model given those columns would
recover our own fabricated tilt, inflating metrics on signal that does not exist
outside this repo -- and it would make the fairness audit circular, measuring the
construction rather than the model.

Holding them out asks the sharper question: does a model trained purely on
BEHAVIOUR perform unevenly across groups anyway? region is the control -- it
carries no generative churn modifier, so its measured spread is the empirical
noise floor against which gaps in the other two are judged.

Any fairness finding here is a statement about this pipeline's ability to DETECT
a gap. It is not evidence about real-world bias, and model_card.json says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# Must precede the pyplot import. Training runs headless -- WSL, containers, CI --
# and the default interactive backend fails at savefig time rather than at import,
# which turns a missing display into a late crash after the model is already fitted.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# CUTOFF / AS_OF / NO_SESSION_RECENCY are the temporal and encoding contract, and
# feature_engineering owns them. Importing rather than redeclaring means the
# model card, the feature spec and the service can never disagree with the
# transform that actually built the training table.
from feature_engineering import AS_OF, CUTOFF, NO_SESSION_RECENCY, build_features

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED = 42

# --- Columns -----------------------------------------------------------------
# The nine behavioural features, in the exact order the model is fitted on. This
# order is contractual: it is written to feature_spec.json and the service must
# assemble its matrix the same way. Reordering silently produces a model that
# scores garbage without raising anything.
FEATURE_COLS = [
    "recency_days",
    "freq_30d",
    "freq_90d",
    "monetary_90d",
    "purchase_count_90d",
    "avg_session_duration",
    "push_open_rate",
    "campaign_click_rate",
    "support_ticket_count",
]

# Fairness audit dimensions. Sliced on, never fitted on (see module docstring).
# is_synthetic is deliberately in neither list: it is constant within a training
# run, and if raw and synthetic customers were ever mixed it would become a
# learnable proxy for which generator produced the row.
SEGMENT_COLS = ["plan_tier", "acquisition_channel", "region"]

TARGET_COL = "churned"

# --- Split -------------------------------------------------------------------
# Stratified on the label only. No temporal split is needed WITHIN training: the
# leakage boundary already sits at CUTOFF and every customer shares that same
# anchor, so there is no second time axis to hold out along.
TEST_SIZE = 0.20

# --- Threshold policy --------------------------------------------------------
# Campaign capacity: the fraction of the customer base contactable per cycle.
# The operating threshold is the predicted-probability quantile that selects
# exactly this share, highest risk first (see module docstring).
TARGET_SELECTION_RATE = 0.30

# --- Logistic-regression preprocessing ---------------------------------------
# XGBoost reads NO_SESSION_RECENCY (999.0) correctly -- it splits the sentinel
# off into its own region, which is exactly the intended "never seen" regime.
# A linear model cannot: it would read 999 as a distance 999 days long and let
# that single value dominate the fitted coefficient, producing a strawman
# baseline that beating proves nothing. So for LR ONLY, the sentinel is split
# into an explicit indicator and the numeric column is capped.
LR_NO_SESSION_COL = "has_no_prior_session"
LR_RECENCY_CAP = 365.0

# --- Explainability ----------------------------------------------------------
# All nine features fit on one SHAP summary plot, so nothing is truncated.
SHAP_MAX_DISPLAY = 9

# --- Paths -------------------------------------------------------------------
DEFAULT_FEATURES_IN = "data/training_features.parquet"
DEFAULT_OUTPUT_DIR = "models"
# The 80 raw customers, held back from training entirely and scored at the end as
# an out-of-distribution check (see main).
DEFAULT_RAW_EVENTS = "data/events.json"


def load_and_split(
    features_path: str | Path,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the feature table, hold out real customers, and split the synthetic set.

    Returns (X_train, X_test, y_train, y_test, X_lr_train, X_lr_test, df_ood).

    Two feature matrices come back because the two model families disagree about
    what NO_SESSION_RECENCY means. X carries the raw 999.0 sentinel for XGBoost,
    which splits it off as its own regime. X_lr caps recency at LR_RECENCY_CAP and
    adds an explicit indicator, because a linear model would otherwise read the
    sentinel as a 999-day distance and let it dominate the coefficient -- see the
    LR_NO_SESSION_COL comment above. Both matrices are sliced with the SAME split
    indices, so row i of X_train and row i of X_lr_train are the same customer.

    Real (non-synthetic) customers never enter the split. They come back untouched
    in df_ood as an out-of-distribution sanity check, so the model is never fitted
    or thresholded on the 80 raw customers it will later be sanity-checked against.
    """
    features_path = Path(features_path)
    if not features_path.exists():
        # Deliberately fatal rather than regenerating: training that silently
        # rebuilds its own input can fit a different dataset than the one the
        # reported metrics claim, and nothing downstream would reveal it.
        raise FileNotFoundError(
            f"Feature table not found: {features_path}\n"
            f"Generate it first:  python src/generate_synthetic.py --n-customers 8000\n"
            f"(training never regenerates its own input -- see load_and_split)"
        )

    df = pd.read_parquet(features_path)
    print(f"Loaded {len(df)} customers from {features_path}")

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    # --- Separate synthetic from real ----------------------------------------
    # .eq(True) rather than a truth test: it is null-safe, so a missing or NaN
    # flag falls to the real side. Real is the conservative default -- a row of
    # unknown provenance must not silently join the training set.
    if "is_synthetic" in df.columns:
        is_synthetic = df["is_synthetic"].eq(True)
    else:
        is_synthetic = pd.Series(False, index=df.index)

    df_synthetic = df[is_synthetic]
    df_ood = df[~is_synthetic]
    print(f"  synthetic (train/test): {len(df_synthetic)}")
    print(f"  real (held out as OOD): {len(df_ood)}")

    if df_synthetic.empty:
        raise ValueError(
            f"No synthetic customers in {features_path} -- nothing to train on.\n"
            f"Expected an is_synthetic=True flag, which build_features merges from "
            f"the customer sidecar. Was this table built without one?"
        )

    # --- Feature matrices (synthetic only) -----------------------------------
    # FEATURE_COLS order is contractual and published in feature_spec.json.
    X = df_synthetic[FEATURE_COLS].astype(float)
    y = df_synthetic[TARGET_COL].astype(int)

    if y.nunique() < 2:
        raise ValueError(f"Training labels are all one class ({y.iloc[0]}) -- cannot fit a classifier.")

    # Exact equality is safe here: build_features assigns NO_SESSION_RECENCY as a
    # literal for customers with no pre-cutoff session, never as a computed value,
    # so there is no floating-point drift to tolerate.
    X_lr = X.copy()
    X_lr[LR_NO_SESSION_COL] = (X_lr["recency_days"] == NO_SESSION_RECENCY).astype(float)
    X_lr["recency_days"] = X_lr["recency_days"].clip(upper=LR_RECENCY_CAP)

    n_sentinel = int(X_lr[LR_NO_SESSION_COL].sum())
    print(f"  no prior session (recency == {NO_SESSION_RECENCY:.0f}): {n_sentinel} "
          f"({n_sentinel / len(X_lr):.1%}) -- raw for XGBoost, split into "
          f"{LR_NO_SESSION_COL} + cap {LR_RECENCY_CAP:.0f} for LR")

    # --- Stratified split ----------------------------------------------------
    # Split the INDEX once and slice both matrices with it, rather than calling
    # train_test_split twice and trusting two calls to agree. They would agree
    # today; they would stop agreeing the moment either call's arguments drifted,
    # and the resulting row misalignment between X and X_lr would be invisible.
    train_idx, test_idx = train_test_split(
        df_synthetic.index, test_size=test_size, random_state=seed, stratify=y
    )

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    X_lr_train, X_lr_test = X_lr.loc[train_idx], X_lr.loc[test_idx]

    # --- Class balance -------------------------------------------------------
    # Printed with the majority-class rate alongside, because that rate is the
    # accuracy floor every headline number has to be read against (see docstring).
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test (test_size={test_size}, seed={seed})")
    for name, split in (("train", y_train), ("test", y_test)):
        churn_rate = split.mean()
        print(f"  {name:<5} n={len(split):<5} churned={int(split.sum()):<5} "
              f"({churn_rate:.3f})  retained={int((1 - split).sum()):<5} ({1 - churn_rate:.3f})  "
              f"accuracy floor={max(churn_rate, 1 - churn_rate):.3f}")

    return X_train, X_test, y_train, y_test, X_lr_train, X_lr_test, df_ood


def _metric_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    is_probability: bool = False,
) -> dict:
    """Score one model into a flat, JSON-serialisable dict.

    Every value is cast to a native Python type: numpy scalars survive in-memory
    comparison but json.dump refuses them, and discovering that at artifact-write
    time means re-running training.

    y_score is any continuous ranking score, not necessarily a probability --
    higher must mean more likely to churn. Passing it adds the threshold-free
    metrics; omit it for a model that emits no ranking (the trivial baseline),
    where ROC-AUC would be undefined rather than 0.5.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    block = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "selection_rate": float(np.mean(y_pred)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_auc": None,
        "pr_auc_churn": None,
        "pr_auc_retention": None,
        "brier": None,
    }

    if y_score is not None:
        block["roc_auc"] = float(roc_auc_score(y_true, y_score))
        block["pr_auc_churn"] = float(average_precision_score(y_true, y_score))
        # Retention is the minority class and the harder call (see module
        # docstring). Negating the score flips the ranking for any monotone
        # score, so this works for the recency rule as well as for probabilities.
        block["pr_auc_retention"] = float(average_precision_score(1 - y_true, -y_score))
        if is_probability:
            block["brier"] = float(brier_score_loss(y_true, y_score))

    return block


def _select_top_k(scores: np.ndarray, rate: float) -> tuple[np.ndarray, float, int]:
    """Flag the top `rate` fraction of customers by score. Returns (pred, threshold, k).

    Selection is by RANK, not by comparing against a probability cut-off, so the
    audience size is exactly the campaign budget regardless of how the scores
    happen to be distributed or calibrated. The implied probability threshold is
    returned alongside for the service to apply.

    Caveat worth knowing: when scores tie exactly at the boundary, rank selection
    splits tied customers arbitrarily while a probability cut-off would take all
    of them and overshoot the budget. Ties are vanishingly rare with continuous
    scores, but the two rules are not identical and the service uses the
    threshold, so its realised selection rate can drift slightly from the budget.
    """
    n = len(scores)
    k = int(round(rate * n))
    # Stable sort so ties break by original order rather than unpredictably --
    # the same input must always produce the same audience.
    order = np.argsort(-scores, kind="stable")
    pred = np.zeros(n, dtype=int)
    pred[order[:k]] = 1
    threshold = float(scores[order[k - 1]]) if k > 0 else float("inf")
    return pred, threshold, k


def train_baselines(
    X_train: pd.DataFrame,
    X_lr_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    X_lr_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, Pipeline]:
    """Fit and evaluate the three baselines XGBoost has to beat.

    Returns (results, lr_pipeline): a dict shaped for direct consumption by
    metrics.json, and the fitted scaler+LR pipeline for persisting as a baseline
    artifact. The pipeline is returned as one object rather than a loose scaler
    and model, because the two must never be applied separately at scoring time.

    The three form a ladder of increasing sophistication, and each answers a
    different question about whether the eventual model is worth deploying:

        trivial       what does predicting nothing get you? This is the accuracy
                      floor, and on a 64%-churn dataset it is embarrassingly high.
        recency rule  what does the single strongest signal get you, with no model
                      at all? A rule a stakeholder can hold in their head, and the
                      thing that actually has to be beaten to justify an ML system.
        logistic reg  what does a competent linear model get you? If XGBoost only
                      matches this, ship this instead -- it is simpler to serve,
                      explain, and monitor.

    All three are evaluated on the same test split. The rule baseline's threshold
    is chosen on TRAIN and then frozen, and the LR pipeline is fitted on TRAIN
    only, so neither sees the test set before it is scored.

    Comparability note: the rule and LR are both held to TARGET_SELECTION_RATE, so
    their recall, precision and F2 are directly comparable. The trivial baseline
    contacts everyone by definition and is NOT budget-feasible -- it is reported
    as a floor reference, not as a deployable option, and its recall of 1.0 should
    be read against that.
    """
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    floor = float(max(y_test_np.mean(), 1 - y_test_np.mean()))

    # Contacting TARGET_SELECTION_RATE of a base that is churn_rate churned can
    # reach at most rate/churn_rate of the churners, even with a perfect ranker.
    # Every recall figure below has to be read against this ceiling, or a model
    # operating near-optimally looks like it is failing badly.
    churn_rate = float(y_test_np.mean())
    max_achievable_recall = float(min(1.0, TARGET_SELECTION_RATE / churn_rate)) if churn_rate else 0.0

    results: dict = {
        "test_n": int(len(y_test_np)),
        "test_churn_rate": churn_rate,
        "majority_class_accuracy_floor": floor,
        "selection_rate_policy": TARGET_SELECTION_RATE,
        "max_achievable_recall": max_achievable_recall,
        "baselines": {},
    }

    print("\n" + "=" * 78)
    print("BASELINES")
    print("=" * 78)

    # --- Baseline 1: trivial -------------------------------------------------
    # Predict churn for everyone. Recall is 1.0 by construction and precision is
    # just the base rate, which is exactly why F2 -- which weights recall 4x --
    # flatters this model badly. That is the point: it shows that no single
    # metric survives contact with a degenerate classifier, and that the metrics
    # only mean something read together.
    pred_trivial = np.ones(len(y_test_np), dtype=int)
    results["baselines"]["trivial_always_churn"] = {
        "description": "Predict churn for every customer (majority class).",
        "threshold": 1.0,
        **_metric_block(y_test_np, pred_trivial),
    }
    print("\n[1] Trivial -- always predict churn  (selection rate 1.00: NOT budget-feasible)")
    _print_metrics(results["baselines"]["trivial_always_churn"], floor)

    # --- Baseline 2: recency rule --------------------------------------------
    # Chosen on TRAIN only, then frozen. Choosing on test would pick the threshold
    # that happens to suit the test set and report the result as if it were held
    # out -- the same self-grading error the CUTOFF split exists to prevent.
    #
    # The threshold is the one whose TRAIN selection rate lands closest to the
    # campaign budget, NOT the one maximising F2. Maximising F2 here degenerates:
    # F2 weights recall 4x, and with churn as the majority class no threshold ever
    # beats predicting all-positive, so the sweep returns 0 and this baseline
    # collapses into the trivial one. Matching the budget instead keeps the rule
    # comparable with LR and XGBoost, which are held to the same audience size --
    # recall and precision mean nothing when compared across different audience
    # sizes.
    recency_train = X_train["recency_days"].to_numpy()
    recency_test = X_test["recency_days"].to_numpy()

    candidates = np.arange(0, 401, dtype=float)
    train_selection = np.array([float((recency_train > t).mean()) for t in candidates])
    best_index = int(np.argmin(np.abs(train_selection - TARGET_SELECTION_RATE)))
    best_threshold = float(candidates[best_index])
    train_selection_rate = float(train_selection[best_index])

    pred_rule = (recency_test > best_threshold).astype(int)
    results["baselines"]["recency_rule"] = {
        "description": f"Predict churn when recency_days > {best_threshold:.0f} days.",
        "threshold": best_threshold,
        "threshold_policy": f"train selection rate closest to {TARGET_SELECTION_RATE:.0%}",
        "train_selection_rate": train_selection_rate,
        # recency_days doubles as the ranking score: more days silent, more risk.
        # NO_SESSION_RECENCY (999) sits above every candidate threshold, so
        # never-seen customers are always flagged -- the intended behaviour.
        **_metric_block(y_test_np, pred_rule, y_score=recency_test),
    }
    print(f"\n[2] Recency rule -- churn when recency_days > {best_threshold:.0f} days "
          f"(chosen on train, train selection rate {train_selection_rate:.3f})")
    _print_metrics(results["baselines"]["recency_rule"], floor, max_achievable_recall)

    # --- Baseline 3: logistic regression -------------------------------------
    # Scaling is required, not cosmetic: recency spans 0-365 while push_open_rate
    # spans 0-1, and an unscaled fit would let the large-magnitude column dominate
    # the L2 penalty and distort every coefficient.
    #
    # class_weight="balanced" downweights churn here, since churn is the MAJORITY
    # class -- the opposite of the usual effect. That distorts LR's predicted
    # probabilities, but the top-K rule below reads only their ORDER, and
    # reweighting a linear model mostly shifts the intercept. Harmless for
    # selection; it would matter if these probabilities were reported as
    # calibrated risk, which they are not.
    # Scaler and model are bound into one Pipeline so they can only ever be
    # applied together. A loose scaler is the classic serving bug: the model
    # loads, scores, and returns confident nonsense when the transform is
    # forgotten or applied in the wrong order.
    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )
    # Fitted on the DataFrame rather than an array so feature names are recorded
    # on the pipeline, letting it detect column drift at scoring time.
    lr_pipeline.fit(X_lr_train, y_train_np)
    scores_lr = lr_pipeline.predict_proba(X_lr_test)[:, 1]

    pred_lr, lr_threshold, k = _select_top_k(scores_lr, TARGET_SELECTION_RATE)
    results["baselines"]["logistic_regression"] = {
        "description": (
            f"L2 logistic regression on {len(X_lr_train.columns)} features "
            f"(scaled, class_weight=balanced); audience = top "
            f"{TARGET_SELECTION_RATE:.0%} by score."
        ),
        "threshold": lr_threshold,
        "threshold_policy": f"top-{TARGET_SELECTION_RATE:.0%} selection rate",
        "n_selected": int(k),
        "coefficients": {
            col: float(coef)
            for col, coef in zip(X_lr_train.columns, lr_pipeline.named_steps["model"].coef_[0])
        },
        **_metric_block(y_test_np, pred_lr, y_score=scores_lr, is_probability=True),
    }
    print(f"\n[3] Logistic regression -- audience = top {TARGET_SELECTION_RATE:.0%} "
          f"({k} of {len(y_test_np)}), implied p >= {lr_threshold:.4f}")
    _print_metrics(results["baselines"]["logistic_regression"], floor, max_achievable_recall)

    _print_comparison(results["baselines"], floor, max_achievable_recall)
    return results, lr_pipeline


def _print_metrics(block: dict, floor: float, max_recall: float | None = None) -> None:
    """Print one metric block, always showing accuracy against the majority floor."""
    delta = block["accuracy"] - floor
    print(f"    accuracy   {block['accuracy']:.4f}   (floor {floor:.4f}, {delta:+.4f})")
    # Recall is quoted as a share of what the budget makes reachable at all; the
    # raw figure alone reads as failure for a model operating near-optimally.
    if max_recall:
        pct = f"  [{block['recall'] / max_recall:.0%} of max {max_recall:.4f}]"
    else:
        pct = ""
    print(f"    recall     {block['recall']:.4f}{pct}")
    print(f"    precision  {block['precision']:.4f}      F2 {block['f2']:.4f}")
    if block["roc_auc"] is not None:
        brier = f"   brier {block['brier']:.4f}" if block["brier"] is not None else ""
        print(f"    ROC-AUC    {block['roc_auc']:.4f}      PR-AUC churn {block['pr_auc_churn']:.4f}"
              f"      PR-AUC retention {block['pr_auc_retention']:.4f}{brier}")
    else:
        print("    ROC-AUC    n/a  (constant score -- no ranking to evaluate)")
    c = block["confusion"]
    print(f"    confusion  TN {c['tn']:<5} FP {c['fp']:<5} FN {c['fn']:<5} TP {c['tp']:<5}"
          f"  selection rate {block['selection_rate']:.3f}")


def _print_comparison(blocks: dict, floor: float, max_recall: float) -> None:
    """Side-by-side table, ordered as the ladder the eventual model must climb."""
    print("\n" + "-" * 92)
    print(f"Recall (max achievable at {TARGET_SELECTION_RATE:.0%} budget: {max_recall:.3f})")
    print("-" * 92)
    print(f"{'model':<24} {'sel':>6} {'acc':>7} {'recall':>8} {'%max':>6} {'prec':>7} "
          f"{'F2':>7} {'ROC-AUC':>8} {'PR-ret':>8} {'FN':>6}")
    print("-" * 92)
    for name, block in blocks.items():
        roc = f"{block['roc_auc']:.4f}" if block["roc_auc"] is not None else "n/a"
        prr = f"{block['pr_auc_retention']:.4f}" if block["pr_auc_retention"] is not None else "n/a"
        pct = f"{block['recall'] / max_recall:.0%}" if max_recall else "n/a"
        print(f"{name:<24} {block['selection_rate']:>6.3f} {block['accuracy']:>7.4f} "
              f"{block['recall']:>8.4f} {pct:>6} {block['precision']:>7.4f} {block['f2']:>7.4f} "
              f"{roc:>8} {prr:>8} {block['confusion']['fn']:>6}")
    print("-" * 92)
    print(f"{'majority-class floor':<24} {'1.000':>6} {floor:>7.4f}"
          f"   <- accuracy below this is worse than guessing")
    print(f"\nRows at sel={TARGET_SELECTION_RATE:.2f} are directly comparable. The trivial baseline "
          f"contacts everyone\n(sel=1.00) and is not budget-feasible -- its recall of 1.0 is free, "
          f"and it is shown only\nas the accuracy floor. %max is recall as a share of what "
          f"{TARGET_SELECTION_RATE:.0%} coverage can reach.")


def _threshold_at_selection_rate(scores: np.ndarray, rate: float) -> tuple[float, float]:
    """Score cut-off that selects `rate` of the population. Returns (threshold, realised rate).

    Used to freeze an operating threshold on TRAIN scores. Unlike _select_top_k,
    which takes exactly k rows and is therefore only definable on the set it is
    applied to, this returns a fixed number the service can apply to any incoming
    batch -- which is what a deployed threshold has to be.

    The realised rate is returned because ties mean the cut-off will not land on
    the target exactly, and because the caller should report what the threshold
    actually did rather than assume it hit the budget.
    """
    threshold = float(np.quantile(scores, 1.0 - rate))
    realised = float(np.mean(scores >= threshold))
    return threshold, realised


def train_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int = RANDOM_SEED,
    baseline_results: dict | None = None,
) -> tuple[xgb.XGBClassifier, float, dict]:
    """Fit the candidate model, freeze its operating threshold, and evaluate it.

    Returns (model, threshold, metrics).

    baseline_results is the dict from train_baselines. It is optional only so this
    function can be run standalone; pass it to get the four-model comparison table
    and the "what does XGBoost buy" verdict, which is the question this section
    exists to answer.

    Hyperparameters, and why each:

        n_estimators=300 + learning_rate=0.05
            Many shallow steps rather than few large ones. A low learning rate
            with more rounds is the standard variance-reduction trade for tabular
            data: each tree corrects a little, so no single split dominates.
        max_depth=4
            Deliberately shallow. Nine features on 4000 rows overfits fast at
            depth 6+, and depth 4 still captures the interactions that matter
            (recency x frequency). It also keeps SHAP attributions readable,
            which is a deliverable here, not a nicety.
        subsample=0.8, colsample_bytree=0.8
            Each tree sees 80% of rows and 80% of columns, decorrelating the
            ensemble so it cannot lean entirely on recency -- the feature that
            already carries most of the signal.
        scale_pos_weight=1.0
            Set EXPLICITLY, because the default reflex is wrong here. Churn is the
            MAJORITY class (~64%), so the usual neg/pos upweighting would push an
            already-dominant class harder, distorting the predicted probabilities
            that the threshold policy and the Brier score both read. 1.0 leaves
            them undistorted. See the module docstring.
        eval_metric="logloss", early_stopping_rounds=20
            Stop once held-out logloss stops improving for 20 rounds, so the
            300-tree budget is a ceiling rather than a target.

    Early stopping runs against a validation split carved out of TRAIN, never
    against the test set. Stopping on test would choose the tree count using the
    same rows the metrics are then reported on -- self-grading, and the same error
    the CUTOFF boundary and the rule baseline's train-only sweep exist to prevent.
    The test set is untouched until final evaluation.
    """
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    floor = float(max(y_test_np.mean(), 1 - y_test_np.mean()))
    churn_rate = float(y_test_np.mean())
    max_achievable_recall = float(min(1.0, TARGET_SELECTION_RATE / churn_rate)) if churn_rate else 0.0

    print("\n" + "=" * 78)
    print("XGBOOST")
    print("=" * 78)

    # Carve the early-stopping validation set out of TRAIN. The model is fitted on
    # X_fit only; X_val decides when to stop. Test is not referenced until the
    # evaluation block below.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=seed, stratify=y_train
    )
    print(f"Fit on {len(X_fit)} rows, early stopping on {len(X_val)} held out of train "
          f"(test set untouched: {len(X_test)} rows)")

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        # Churn is the majority class here -- see docstring above. Do not "fix"
        # this to neg/pos without re-reading that paragraph.
        scale_pos_weight=1.0,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit.to_numpy(), eval_set=[(X_val, y_val.to_numpy())], verbose=False)

    best_iteration = int(getattr(model, "best_iteration", model.n_estimators - 1))
    print(f"Trained: best_iteration={best_iteration} of {model.n_estimators} "
          f"(early stopping after 20 rounds without improvement on val)")

    # --- Freeze the operating threshold --------------------------------------
    # Derived on the VALIDATION split, then applied unchanged to test, so the test
    # selection rate is an OUTCOME we observe rather than a constraint we impose.
    #
    # Val rather than the full train set: the model was fitted on X_fit, so its
    # scores there are optimistically extreme, and a quantile taken from them
    # misplaces the cut-off. Val was only used for the stopping decision, so its
    # score distribution is a near-unbiased stand-in for unseen data -- which is
    # what a threshold has to generalise to. Deriving it from test would fit the
    # threshold to the data it is scored on and manufacture a 30% selection rate
    # that production could never reproduce.
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold, train_selection = _threshold_at_selection_rate(val_scores, TARGET_SELECTION_RATE)

    test_scores = model.predict_proba(X_test)[:, 1]
    pred = (test_scores >= threshold).astype(int)

    print(f"Threshold frozen on val: p >= {threshold:.4f} "
          f"(val selection rate {train_selection:.3f}) -> test selection rate {pred.mean():.3f}")

    block = {
        "description": (
            f"XGBoost ({best_iteration + 1} trees, depth 4, lr 0.05); "
            f"threshold frozen on validation at {TARGET_SELECTION_RATE:.0%} selection."
        ),
        "threshold": threshold,
        "threshold_policy": f"validation selection rate closest to {TARGET_SELECTION_RATE:.0%}, frozen",
        "val_selection_rate": train_selection,
        "n_fit": int(len(X_fit)),
        "n_val": int(len(X_val)),
        "best_iteration": best_iteration,
        "n_estimators_budget": int(model.n_estimators),
        "max_achievable_recall": max_achievable_recall,
        **_metric_block(y_test_np, pred, y_score=test_scores, is_probability=True),
    }
    block["pct_max_recall"] = float(block["recall"] / max_achievable_recall) if max_achievable_recall else None

    _print_metrics(block, floor, max_achievable_recall)

    metrics = {
        "test_n": int(len(y_test_np)),
        "test_churn_rate": churn_rate,
        "majority_class_accuracy_floor": floor,
        "selection_rate_policy": TARGET_SELECTION_RATE,
        "max_achievable_recall": max_achievable_recall,
        "xgboost": block,
    }

    # --- Four-model comparison ----------------------------------------------
    if baseline_results:
        combined = {**baseline_results["baselines"], "xgboost": block}
        _print_comparison(combined, floor, max_achievable_recall)
        matched = _print_verdict(
            block, baseline_results["baselines"], test_scores, X_test["recency_days"].to_numpy(), y_test_np
        )
        block["matched_audience_comparison"] = matched

    return model, threshold, metrics


def _matched_audience(scores: np.ndarray, y_true: np.ndarray, rate: float) -> dict:
    """Score a ranking at an exactly-matched audience size, so models are comparable.

    Frozen thresholds land each model on a slightly different realised selection
    rate, and false-negative counts are not comparable across different audience
    sizes: contacting more people finds more churners for free. This re-selects
    every model's top `rate` fraction by rank so the only thing that differs is
    the quality of the ordering.
    """
    n = len(scores)
    k = int(round(rate * n))
    order = np.argsort(-scores, kind="stable")
    pred = np.zeros(n, dtype=int)
    pred[order[:k]] = 1
    return {
        "selection_rate": float(pred.mean()),
        "n_selected": int(k),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "false_negatives": int(((y_true == 1) & (pred == 0)).sum()),
        "true_positives": int(((y_true == 1) & (pred == 1)).sum()),
    }


def _print_verdict(
    xgb_block: dict,
    baselines: dict,
    xgb_scores: np.ndarray,
    rule_scores: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """State plainly what XGBoost buys over the rule -- the question Section 4 exists to answer.

    Compared against the recency rule specifically, not against the trivial
    baseline: the rule is the honest bar, because it is free to build, trivial to
    explain, and needs no serving infrastructure at all. If the model cannot beat
    a threshold on one column, the model is not worth the pipeline behind it.

    Ranking metrics lead, because they are the only part of the comparison that
    does not depend on how many customers each model happened to select. The
    false-negative comparison follows, at an explicitly matched audience.
    """
    rule = baselines.get("recency_rule")
    if not rule:
        return {}

    roc_delta = xgb_block["roc_auc"] - rule["roc_auc"]
    pr_ret_delta = xgb_block["pr_auc_retention"] - rule["pr_auc_retention"]

    xgb_matched = _matched_audience(xgb_scores, y_true, TARGET_SELECTION_RATE)
    rule_matched = _matched_audience(rule_scores, y_true, TARGET_SELECTION_RATE)
    found_delta = xgb_matched["true_positives"] - rule_matched["true_positives"]

    print("\nVERDICT -- what XGBoost buys over the recency rule:")
    print(f"  Ranking (independent of audience size): ROC-AUC {roc_delta:+.4f} "
          f"({rule['roc_auc']:.4f} -> {xgb_block['roc_auc']:.4f}), "
          f"PR-AUC-retention {pr_ret_delta:+.4f} "
          f"({rule['pr_auc_retention']:.4f} -> {xgb_block['pr_auc_retention']:.4f}).")

    verb = "more" if found_delta >= 0 else "fewer"
    print(f"  At an equal {TARGET_SELECTION_RATE:.0%} budget ({xgb_matched['n_selected']} contacted): "
          f"XGBoost finds {abs(found_delta)} {verb} churners than the rule "
          f"({xgb_matched['true_positives']} vs {rule_matched['true_positives']} caught; "
          f"FN {xgb_matched['false_negatives']} vs {rule_matched['false_negatives']}).")

    if found_delta <= 0:
        # Said plainly rather than buried: a model that does not beat the rule at
        # the operating point is a finding, not a failure to report around.
        print("  At this budget the better ranking does NOT convert into more churners caught. "
              "It would\n  pay off at a different budget, or for tiering the audience -- not here.")

    return {"xgboost": xgb_matched, "recency_rule": rule_matched, "churners_found_delta": int(found_delta)}


# --- Explainability copy -----------------------------------------------------
# One business sentence per feature, describing what a POSITIVE SHAP value means
# -- i.e. what pushes this customer's churn score UP. Written for a product
# stakeholder, not a modeller: no mention of SHAP, log-odds or features.
#
# `expected` is the direction we believe the business logic implies. It is not
# used to constrain the model; it is checked against what the model actually
# learned, so a sign flip surfaces as a finding rather than hiding inside a plot.
FEATURE_INTERPRETATIONS = {
    "recency_days": {
        "expected": "positive",
        "meaning": "The longer a customer has gone without opening the app, the higher their churn risk -- silence is the earliest and strongest warning.",
    },
    "freq_30d": {
        "expected": "negative",
        "meaning": "Opening the app repeatedly in the last month signals an intact habit, and habit is what keeps customers next month.",
    },
    "freq_90d": {
        "expected": "negative",
        "meaning": "A steady three-month visit history separates a genuine regular from someone who showed up once; regulars churn less.",
    },
    "monetary_90d": {
        "expected": "negative",
        "meaning": "Money already spent creates switching friction -- paying customers have something to lose by leaving.",
    },
    "purchase_count_90d": {
        "expected": "negative",
        "meaning": "Buying repeatedly matters more than buying big: a transacting habit shows ongoing intent, not a single impulse.",
    },
    "avg_session_duration": {
        "expected": "negative",
        "meaning": "Short visits mean shallow engagement; dwell time falling off is an early warning that arrives before outright silence.",
    },
    "push_open_rate": {
        "expected": "negative",
        "meaning": "Ignoring notifications shows the app has lost mindshare -- reachability decays before usage does.",
    },
    "campaign_click_rate": {
        "expected": "negative",
        "meaning": "Clicking through to content is stronger intent than merely opening a push; near-zero means marketing can no longer pull this customer back.",
    },
    "support_ticket_count": {
        "expected": "positive",
        "meaning": "Support tickets are logged dissatisfaction -- unresolved billing disputes and bugs are a leading cause of customers leaving voluntarily.",
    },
}


def compute_shap(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    output_dir: str | Path,
    threshold: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Explain the fitted model with SHAP and write the explainability artifacts.

    Returns (shap_values, importance_df).

    `threshold` is the frozen operating threshold, used to pick the boundary
    customer -- the one sitting closest to the in/out line, whose explanation is
    the one a stakeholder disputing an audience decision will actually ask about.
    It defaults to the median score if not supplied, so the function can be run
    standalone.

    TreeExplainer is exact for gradient-boosted trees: it walks the tree structure
    rather than sampling perturbations, so there is no approximation error and no
    background dataset to choose. SHAP values are in LOG-ODDS, and they are
    additive -- base value plus every feature contribution reconstructs the
    model's raw output for that customer exactly. That additivity is what makes
    the waterfall plots defensible in a stakeholder conversation: the numbers on
    the chart sum to the score.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("SHAP EXPLAINABILITY")
    print("=" * 78)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)
    shap_values = explanation.values

    scores = model.predict_proba(X_test)[:, 1]
    if threshold is None:
        threshold = float(np.median(scores))
        print(f"No threshold supplied -- using median score {threshold:.4f} for the boundary case.")

    # --- Global importance ---------------------------------------------------
    mean_abs = np.abs(shap_values).mean(axis=0)
    # Sign of the value/attribution relationship: does a HIGH feature value push
    # the score up or down? Computed from the model's own attributions rather
    # than assumed, so the check below is meaningful.
    directions = []
    for j, col in enumerate(X_test.columns):
        feature_values = X_test.iloc[:, j].to_numpy()
        if np.std(feature_values) == 0 or np.std(shap_values[:, j]) == 0:
            directions.append(np.nan)
        else:
            directions.append(float(np.corrcoef(feature_values, shap_values[:, j])[0, 1]))

    importance_df = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "mean_abs_shap": mean_abs,
            "value_shap_corr": directions,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    importance_df["rank"] = importance_df.index + 1
    importance_df["direction"] = np.where(
        importance_df["value_shap_corr"].isna(), "flat",
        np.where(importance_df["value_shap_corr"] > 0, "positive", "negative"),
    )

    # --- Plots ---------------------------------------------------------------
    shap.plots.bar(explanation, max_display=SHAP_MAX_DISPLAY, show=False)
    _save_plot(output_dir / "shap_importance.png")

    shap.plots.beeswarm(explanation, max_display=SHAP_MAX_DISPLAY, show=False)
    _save_plot(output_dir / "shap_beeswarm.png")

    # --- Three individual explanations ---------------------------------------
    # Highest and lowest are the easy cases -- useful for showing the model's
    # reasoning at the extremes. The boundary case is the important one: it is
    # the marginal customer the campaign either does or does not contact.
    cases = {
        "high": int(np.argmax(scores)),
        "low": int(np.argmin(scores)),
        "boundary": int(np.argmin(np.abs(scores - threshold))),
    }

    case_details = {}
    for label, position in cases.items():
        shap.plots.waterfall(explanation[position], max_display=SHAP_MAX_DISPLAY, show=False)
        _save_plot(output_dir / f"shap_waterfall_{label}.png")
        contributions = dict(zip(X_test.columns, shap_values[position]))
        top_driver = max(contributions, key=lambda c: abs(contributions[c]))
        case_details[label] = {
            "row_position": position,
            "customer_index": int(X_test.index[position]),
            "churn_score": float(scores[position]),
            "top_driver": top_driver,
            "top_driver_shap": float(contributions[top_driver]),
            "feature_values": {c: float(v) for c, v in X_test.iloc[position].items()},
        }

    # --- Interpretations -----------------------------------------------------
    # Each entry pairs the business sentence with what the model actually did, so
    # a reader can see where the two disagree instead of trusting the copy.
    interpretations = {
        "base_value_log_odds": float(np.mean(explanation.base_values)),
        "units": "SHAP values are log-odds contributions; positive pushes churn risk up.",
        "features": {},
        "cases": case_details,
    }
    contradictions = []
    for row in importance_df.itertuples():
        spec = FEATURE_INTERPRETATIONS.get(row.feature, {})
        expected = spec.get("expected")
        matches = (expected == row.direction) if expected and row.direction != "flat" else None
        if matches is False:
            contradictions.append(row.feature)
        interpretations["features"][row.feature] = {
            "rank": int(row.rank),
            "mean_abs_shap": float(row.mean_abs_shap),
            "observed_direction": row.direction,
            "expected_direction": expected,
            "matches_expectation": matches,
            "meaning": spec.get("meaning", "No interpretation recorded for this feature."),
        }

    interpretations["direction_contradictions"] = contradictions

    with (output_dir / "shap_interpretations.json").open("w") as fh:
        json.dump(interpretations, fh, indent=2)

    # --- Report --------------------------------------------------------------
    print(f"\nBase value (log-odds): {interpretations['base_value_log_odds']:+.4f}")
    print("\nTop 3 drivers by mean |SHAP|:")
    for row in importance_df.head(3).itertuples():
        spec = FEATURE_INTERPRETATIONS.get(row.feature, {})
        print(f"  {row.rank}. {row.feature:<22} mean|SHAP| = {row.mean_abs_shap:.4f}   "
              f"({row.direction}: {'higher value -> more churn risk' if row.direction == 'positive' else 'higher value -> less churn risk'})")
        print(f"     {spec.get('meaning', '')}")

    if contradictions:
        # Not buried in the JSON: a feature the model reads backwards from the
        # business story is either a data problem or a story problem, and either
        # way it must not reach a stakeholder deck unexamined.
        print(f"\n  WARNING: {len(contradictions)} feature(s) contradict the expected direction: "
              f"{', '.join(contradictions)}")
        print("  Check these before presenting -- the model disagrees with the business rationale.")

    print(f"\nWrote: shap_importance.png, shap_beeswarm.png, "
          f"shap_waterfall_{{high,low,boundary}}.png, shap_interpretations.json -> {output_dir}")

    return shap_values, importance_df


def _save_plot(path: Path) -> None:
    """Save the current figure and close it, so figures cannot accumulate."""
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


# --- Fairness ----------------------------------------------------------------
# The control dimension: SEGMENT_CHURN_TARGETS in generate_synthetic assigns it no
# churn modifier, so whatever FNR spread it shows is pure noise. That measured
# spread becomes the yardstick for every other dimension -- far more defensible
# than an arbitrary rule of thumb like the 80% rule, because it is calibrated on
# this model, this sample size and this threshold.
FAIRNESS_CONTROL_COL = "region"

# 95% two-sided normal quantile, for the Wilson score interval.
WILSON_Z = 1.959963985


def _wilson_interval(successes: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the textbook normal approximation because the group counts
    here are small and the rates are far from 0.5 -- exactly where the normal
    interval misbehaves, running past 0 or 1 and understating uncertainty. Wilson
    stays inside [0, 1] and holds its coverage at small n, which is the whole
    point when a subgroup cell has 40 churners in it.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = (z / denominator) * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return (max(0.0, center - half), min(1.0, center + half))


def fairness_audit(
    model: xgb.XGBClassifier,
    threshold: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test_segments: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, float]:
    """Audit false-negative rates across synthetic segments. Returns (fairness_df, noise_floor).

    Criterion: EQUAL OPPORTUNITY -- equal FNR (equivalently, equal recall) across
    groups. Justified in the printed output and in fairness.json.

    Measured at the frozen operating threshold, not on the raw scores, because a
    customer either receives the retention campaign or does not. A model can rank
    well in aggregate and still miss one group disproportionately once a single
    cut-off is applied to everyone, and it is the cut-off that determines who is
    actually helped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Align segments to the test rows. Reindex rather than assume: a silent row
    # misalignment here would attribute each customer's outcome to the wrong
    # group and produce a confident, entirely fictional fairness finding.
    segments = df_test_segments.reindex(X_test.index)
    missing_segments = int(segments[SEGMENT_COLS].isna().all(axis=1).sum())
    if missing_segments:
        print(f"  NOTE: {missing_segments} test rows have no segment data and are excluded per-slice.")

    y_true = y_test.to_numpy()
    scores = model.predict_proba(X_test)[:, 1]
    pred = (scores >= threshold).astype(int)

    rows: list[dict] = []
    for segment_col in SEGMENT_COLS:
        if segment_col not in segments.columns:
            continue
        values = segments[segment_col]
        for group in sorted(v for v in values.dropna().unique()):
            mask = (values == group).to_numpy()
            g_true, g_pred = y_true[mask], pred[mask]

            tp = int(((g_true == 1) & (g_pred == 1)).sum())
            fn = int(((g_true == 1) & (g_pred == 0)).sum())
            fp = int(((g_true == 0) & (g_pred == 1)).sum())
            tn = int(((g_true == 0) & (g_pred == 0)).sum())

            churners = tp + fn
            retained = fp + tn
            # FNR: of the churners in this group, what share did we fail to
            # contact? This is the harm metric -- see the criterion note below.
            fnr = fn / churners if churners else float("nan")
            fnr_lo, fnr_hi = _wilson_interval(fn, churners)

            # Counterfactual FNR if this group were contacted at exactly the
            # campaign rate instead of sharing one global cut-off. The difference
            # between the two isolates how much of any gap is base-rate
            # arithmetic and how much is the model treating groups differently.
            g_scores = scores[mask]
            k_group = int(round(TARGET_SELECTION_RATE * int(mask.sum())))
            eq_pred = np.zeros(int(mask.sum()), dtype=int)
            eq_pred[np.argsort(-g_scores, kind="stable")[:k_group]] = 1
            fn_eq = int(((g_true == 1) & (eq_pred == 0)).sum())
            fnr_eq = fn_eq / churners if churners else float("nan")

            # Within-group ranking quality. This is the ONLY fully base-rate-free
            # measure available: it asks whether the model orders this group's
            # churners above its non-churners as well as it does for any other
            # group, and it is unaffected by how many churners the group contains
            # or how many of them the budget can reach. FNR -- even at equal
            # selection -- cannot do this, because the minimum achievable FNR is
            # itself a function of base rate (1 - rate/base_rate).
            if churners and retained:
                auc_within = float(roc_auc_score(g_true, g_scores))
                ceiling = min(1.0, TARGET_SELECTION_RATE / g_true.mean())
                recall_vs_ceiling = float((1.0 - fnr_eq) / ceiling) if ceiling else float("nan")
            else:
                auc_within = float("nan")
                recall_vs_ceiling = float("nan")

            rows.append({
                "segment": segment_col,
                "group": group,
                "n": int(mask.sum()),
                "n_churners": churners,
                "base_rate": float(g_true.mean()) if mask.sum() else float("nan"),
                "fnr": fnr,
                "fnr_ci_low": fnr_lo,
                "fnr_ci_high": fnr_hi,
                "fnr_equal_selection": fnr_eq,
                "auc_within_group": auc_within,
                "recall_vs_ceiling": recall_vs_ceiling,
                "fpr": fp / retained if retained else float("nan"),
                "selection_rate": float(g_pred.mean()) if mask.sum() else float("nan"),
                "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })

    fairness_df = pd.DataFrame(rows)

    # --- Noise floor ---------------------------------------------------------
    control = fairness_df[fairness_df["segment"] == FAIRNESS_CONTROL_COL]
    noise_floor = float(control["fnr"].max() - control["fnr"].min()) if not control.empty else float("nan")
    # Separate floor for the equal-selection view: a spread measured under
    # per-group thresholds must be judged against a control measured the same
    # way, or the comparison mixes two different sampling regimes.
    noise_floor_equal = (
        float(control["fnr_equal_selection"].max() - control["fnr_equal_selection"].min())
        if not control.empty else float("nan")
    )
    # The bias yardstick. Residual bias is judged on within-group AUC spread
    # against this, NOT on either FNR spread -- both of those stay confounded by
    # base rate (see the auc_within_group comment above).
    control_auc_spread = (
        float(control["auc_within_group"].max() - control["auc_within_group"].min())
        if not control.empty else float("nan")
    )

    print("\n" + "=" * 78)
    print("FAIRNESS AUDIT")
    print("=" * 78)
    print(f"Criterion: EQUAL OPPORTUNITY (equal FNR across groups), at threshold p >= {threshold:.4f}.")
    print(f"Noise floor ('{FAIRNESS_CONTROL_COL}' control): {noise_floor:.4f} FNR spread at global threshold.")
    print(f"Bias metric: within-group ROC-AUC; control spread {control_auc_spread:.4f}, "
          f"flag above 2x = {2 * control_auc_spread:.4f}.")

    # Control first: the yardstick has to exist before anything is measured
    # against it, so the tables are ordered control-first rather than in
    # SEGMENT_COLS order.
    ordered = [FAIRNESS_CONTROL_COL] + [c for c in SEGMENT_COLS if c != FAIRNESS_CONTROL_COL]
    for segment_col in ordered:
        block = fairness_df[fairness_df["segment"] == segment_col]
        if block.empty:
            continue
        block = block.sort_values("fnr", ascending=False)

        print(f"\n--- {segment_col} ---")
        print(f"{'group':<14} {'n':>5} {'churn':>6} {'base':>7} {'FNR':>7} {'FNR 95% CI':>16} "
              f"{'FNR@eq':>7} {'AUC':>7} {'FPR':>7} {'sel':>7} {'prec':>7}")
        for row in block.itertuples():
            ci = f"[{row.fnr_ci_low:.3f}, {row.fnr_ci_high:.3f}]"
            print(f"{str(row.group):<14} {row.n:>5} {row.n_churners:>6} {row.base_rate:>7.3f} "
                  f"{row.fnr:>7.3f} {ci:>16} {row.fnr_equal_selection:>7.3f} "
                  f"{row.auc_within_group:>7.4f} {row.fpr:>7.3f} "
                  f"{row.selection_rate:>7.3f} {row.precision:>7.3f}")

        _print_fairness_interpretation(segment_col, block, noise_floor, control_auc_spread)

    _print_impossibility_note(fairness_df, noise_floor, noise_floor_equal)
    _print_fairness_criterion()
    _print_fairness_caveats(fairness_df, noise_floor)

    payload = {
        "criterion": "equal_opportunity",
        "criterion_definition": "Equal false-negative rate (equivalently, equal recall) across groups.",
        "criterion_rationale": (
            "The score selects who receives a retention intervention. A false negative is a "
            "churner who is never contacted and leaves -- the cost is that customer's lifetime "
            "value, and it is invisible because the counterfactual is never observed. A false "
            "positive is a retained customer who receives an offer they did not need, costing "
            "marginal campaign spend. Unequal FNR therefore means one group is systematically "
            "denied access to the intervention. Demographic parity (equal selection rate) is "
            "rejected: base churn rates genuinely differ across these groups, so equalising "
            "selection would over-contact low-churn groups and under-contact high-churn ones, "
            "making outcomes worse for the group already churning most."
        ),
        "bias_metric": "within-group ROC-AUC vs region control",
        "bias_metric_rationale": (
            "FNR is confounded by base rate at a global threshold (through the selection rate) "
            "and even at equal selection (through the achievable ceiling, min FNR = "
            "1 - rate/base_rate). Within-group AUC is free of both. Verdict threshold is 2x the "
            "control's own AUC spread."
        ),
        "control_auc_spread": control_auc_spread,
        "bias_flag_threshold": 2.0 * control_auc_spread,
        "chouldechova_note": "FNR gaps are base-rate-explained, not model bias",
        "impossibility_result": (
            "Chouldechova 2017 -- calibration and equal_opportunity cannot simultaneously hold "
            "with unequal base rates. Model optimized for calibration."
        ),
        "plan_tier_fnr_global": (
            float(fairness_df[fairness_df["segment"] == "plan_tier"]["fnr"].max()
                  - fairness_df[fairness_df["segment"] == "plan_tier"]["fnr"].min())
            if "plan_tier" in set(fairness_df["segment"]) else None
        ),
        "plan_tier_fnr_equal_selection": (
            float(fairness_df[fairness_df["segment"] == "plan_tier"]["fnr_equal_selection"].max()
                  - fairness_df[fairness_df["segment"] == "plan_tier"]["fnr_equal_selection"].min())
            if "plan_tier" in set(fairness_df["segment"]) else None
        ),
        "noise_floor_region": noise_floor,
        "threshold": float(threshold),
        "noise_floor_fnr_spread": noise_floor,
        "noise_floor_fnr_spread_equal_selection": noise_floor_equal,
        "noise_floor_source": FAIRNESS_CONTROL_COL,
        "caveats": [
            "Segments are SYNTHETIC (invented in generate_synthetic). Findings describe this "
            "pipeline's ability to DETECT a subgroup gap, not real-world bias.",
            f"'{FAIRNESS_CONTROL_COL}' carries no generative churn modifier and is the control: "
            f"its FNR spread of {noise_floor:.4f} is the empirical noise floor.",
            "acquisition_channel may be underpowered: its middle groups are separated by less "
            "than the noise floor by construction. Check n_churners per cell before concluding.",
        ],
        "segments": {
            col: {
                "fnr_spread": float(
                    fairness_df[fairness_df["segment"] == col]["fnr"].max()
                    - fairness_df[fairness_df["segment"] == col]["fnr"].min()
                ),
                "fnr_spread_equal_selection": float(
                    fairness_df[fairness_df["segment"] == col]["fnr_equal_selection"].max()
                    - fairness_df[fairness_df["segment"] == col]["fnr_equal_selection"].min()
                ),
                "exceeds_noise_floor": bool(
                    (fairness_df[fairness_df["segment"] == col]["fnr"].max()
                     - fairness_df[fairness_df["segment"] == col]["fnr"].min()) > noise_floor
                ),
                "residual_bias_after_equal_selection": bool(
                    (fairness_df[fairness_df["segment"] == col]["fnr_equal_selection"].max()
                     - fairness_df[fairness_df["segment"] == col]["fnr_equal_selection"].min())
                    > noise_floor_equal
                ),
                "groups": fairness_df[fairness_df["segment"] == col].to_dict(orient="records"),
            }
            for col in fairness_df["segment"].unique()
        },
    }
    # Per-segment verdicts, derived rather than hardcoded: a verdict string that
    # does not move with the data is worse than no verdict at all.
    for segment_col in fairness_df["segment"].unique():
        if segment_col == FAIRNESS_CONTROL_COL:
            continue
        block = fairness_df[fairness_df["segment"] == segment_col]
        verdict, auc_spread, multiple = _bias_verdict(block, control_auc_spread)
        rc_spread = float(block["recall_vs_ceiling"].max() - block["recall_vs_ceiling"].min())
        control_rc = fairness_df[fairness_df["segment"] == FAIRNESS_CONTROL_COL]
        control_rc_spread = float(control_rc["recall_vs_ceiling"].max() - control_rc["recall_vs_ceiling"].min())
        if verdict == "NO RESIDUAL BIAS":
            detail = f"{verdict} -- AUC spread {multiple:.1f}x control, within noise"
        else:
            within = "within" if rc_spread <= control_rc_spread else "above"
            detail = (f"INCONCLUSIVE -- AUC spread {multiple:.1f}x control but recall/ceiling "
                      f"{within} noise")
        payload[f"{segment_col}_bias_verdict"] = detail
        payload["segments"][segment_col]["auc_spread"] = auc_spread
        payload["segments"][segment_col]["auc_spread_vs_control"] = multiple
        payload["segments"][segment_col]["bias_verdict"] = detail

    with (output_dir / "fairness.json").open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote: fairness.json -> {output_dir}")

    return fairness_df, noise_floor


def _bias_verdict(block: pd.DataFrame, control_auc_spread: float) -> tuple[str, float, float]:
    """Judge residual bias on within-group AUC. Returns (verdict, auc_spread, multiple).

    FNR is reported but never used for this verdict. Both FNR views stay
    confounded by base rate -- at a global threshold through the selection rate,
    and even at equal selection through the achievable ceiling
    (min FNR = 1 - rate/base_rate). AUC is free of both: it measures only whether
    the model orders a group's churners above its non-churners.

    The threshold is 2x the control's own spread. The control has no real effect,
    so its spread is what pure sampling noise produces at these cell sizes;
    anything within 2x of that is not separable from noise.
    """
    auc_spread = float(block["auc_within_group"].max() - block["auc_within_group"].min())
    multiple = auc_spread / control_auc_spread if control_auc_spread else float("inf")
    verdict = "NO RESIDUAL BIAS" if multiple <= 2.0 else "FLAG FOR REVIEW"
    return verdict, auc_spread, multiple


def _print_fairness_interpretation(
    segment_col: str, block: pd.DataFrame, noise_floor: float, control_auc_spread: float
) -> None:
    """Report both FNR views for the record, then judge bias on AUC alone.

    The FNR columns stay because the global-threshold FNR is real production harm
    regardless of what causes it: those churners genuinely go uncontacted. What
    changed is that the harm measure no longer doubles as the bias verdict.
    """
    spread = float(block["fnr"].max() - block["fnr"].min())
    spread_equal = float(block["fnr_equal_selection"].max() - block["fnr_equal_selection"].min())
    auc_spread = float(block["auc_within_group"].max() - block["auc_within_group"].min())

    if segment_col == FAIRNESS_CONTROL_COL:
        print(f"  CONTROL. FNR spread {spread:.4f} (global), {spread_equal:.4f} (equal selection); "
              f"within-group AUC spread {auc_spread:.4f}.")
        print(f"  No generative churn modifier here, so these are the empirical NOISE FLOORS. "
              f"Residual bias\n  elsewhere is judged against 2x the AUC spread "
              f"({2 * auc_spread:.4f}).")
        return

    verdict, _, multiple = _bias_verdict(block, control_auc_spread)
    worst_auc = block.loc[block["auc_within_group"].idxmin()]
    best_auc = block.loc[block["auc_within_group"].idxmax()]

    print(f"  FNR gap at global threshold: {spread:.3f} "
          f"({'exceeds' if spread > noise_floor else 'within'} noise floor {noise_floor:.3f}) "
          f"-- production harm, base-rate-driven")
    print(f"  FNR gap at equal selection:  {spread_equal:.3f} (reported only; still "
          f"base-rate-confounded via the FNR ceiling)")
    print(f"  Within-group AUC spread:     {auc_spread:.4f} = {multiple:.1f}x control "
          f"-> {verdict}")

    if verdict == "NO RESIDUAL BIAS":
        print(f"  -> The model ranks every group's churners about equally well "
              f"({worst_auc['group']} {worst_auc['auc_within_group']:.4f} .. "
              f"{best_auc['group']} {best_auc['auc_within_group']:.4f}).")
        print(f"     The FNR gap is base-rate arithmetic under one global cut-off, not bias.")
    else:
        print(f"  -> The model ranks '{worst_auc['group']}' churners worse "
              f"({worst_auc['auc_within_group']:.4f}) than '{best_auc['group']}' "
              f"({best_auc['auc_within_group']:.4f}),")
        print(f"     beyond what the control's noise explains. Investigate before shipping.")


def _print_impossibility_note(fairness_df: pd.DataFrame, noise_floor: float, noise_floor_equal: float) -> None:
    """Name the result driving the plan_tier gap, so it is not mistaken for bias."""
    tier = fairness_df[fairness_df["segment"] == "plan_tier"]
    if tier.empty:
        return
    spread = float(tier["fnr"].max() - tier["fnr"].min())
    spread_equal = float(tier["fnr_equal_selection"].max() - tier["fnr_equal_selection"].min())

    print("\n--- Impossibility result ---")
    print("Calibration and equal FNR cannot simultaneously hold when base rates differ")
    print("(Chouldechova 2017). This model is calibrated. The plan_tier FNR gap is")
    print("arithmetic, not bias. Remedy: per-group thresholds -- business decision.")
    print(f"\n  plan_tier FNR gap: {spread:.3f} at global threshold "
          f"(floor {noise_floor:.3f}), {spread_equal:.3f} at equal selection "
          f"(floor {noise_floor_equal:.3f}).")
    print("  Per-group thresholds would close it, at the cost of contacting lower-risk")
    print("  premium customers ahead of higher-risk free-tier ones -- less efficient use of")
    print("  a fixed budget. Which matters more is a business call, not a modelling one.")


def _print_fairness_criterion() -> None:
    """State the criterion and why the obvious alternative is wrong here."""
    print("\n--- Criterion ---")
    print("EQUAL OPPORTUNITY (equal FNR across groups) is the criterion.")
    print("  A false negative is a churner who is never contacted, leaves, and is never")
    print("  counted -- the cost is their lifetime value. A false positive is a retained")
    print("  customer who gets an offer they did not need -- the cost is marginal campaign")
    print("  spend. Cost(FN) >> cost(FP), so an FNR gap means one group is systematically")
    print("  DENIED the retention offer. That is the harm worth measuring.")
    print("\nDEMOGRAPHIC PARITY (equal selection rate) is rejected, and would be actively")
    print("  harmful here. Base churn rates genuinely differ across these groups. Forcing")
    print("  equal selection would contact low-churn groups beyond what their risk warrants")
    print("  and starve high-churn groups of contact -- making outcomes worse for exactly the")
    print("  group already churning most, in the name of fairness.")


def _print_fairness_caveats(fairness_df: pd.DataFrame, noise_floor: float) -> None:
    """The three caveats, printed rather than left to the JSON nobody opens."""
    channel = fairness_df[fairness_df["segment"] == "acquisition_channel"]
    min_churners = int(channel["n_churners"].min()) if not channel.empty else 0

    print("\n--- Caveats ---")
    print("(a) Segments are SYNTHETIC -- invented in generate_synthetic, with churn")
    print("    associations written in by hand. Every finding here is a statement about this")
    print("    pipeline's ability to DETECT a subgroup gap, not evidence of real-world bias.")
    print(f"(b) '{FAIRNESS_CONTROL_COL}' is the control: no generative churn modifier, so its")
    print(f"    FNR spread of {noise_floor:.4f} is the noise floor, not a finding. A gap in")
    print("    another segment only means something if it clears this.")
    print(f"(c) acquisition_channel is likely UNDERPOWERED: smallest cell has {min_churners}")
    print("    churners, and its middle groups are separated by less than the noise floor by")
    print("    construction. Read its ordering as indicative only; the endpoints are the only")
    print("    contrast the sample size supports.")


def score_ood(model: xgb.XGBClassifier, threshold: float, df_ood: pd.DataFrame,
              raw_events_path: str | Path, synthetic_churn_rate: float) -> dict:
    """Score the held-out raw customers as an out-of-distribution sanity check.

    The 80 raw customers never touch training, the split, the threshold or the
    audit. Scoring them last answers the one question the synthetic pipeline
    cannot answer about itself: does a model fitted entirely on generated data
    behave sensibly on the real events it was modelled from?

    df_ood comes from load_and_split when the feature table happens to contain
    real rows. With a synthetic-only table it is empty, so the raw events are
    transformed here instead -- through the SAME build_features used in training,
    which is the point: any drift between the two paths would show up as nonsense
    predictions rather than hiding.
    """
    print("\n" + "=" * 78)
    print("OUT-OF-DISTRIBUTION CHECK (raw sample)")
    print("=" * 78)

    if df_ood is None or df_ood.empty:
        raw_events_path = Path(raw_events_path)
        if not raw_events_path.exists():
            print(f"  Skipped: {raw_events_path} not found.")
            return {"status": "skipped", "reason": f"{raw_events_path} not found"}
        with raw_events_path.open() as fh:
            raw_events = json.load(fh)
        print(f"  Building features from {len(raw_events)} raw events in {raw_events_path}")
        df_ood = build_features(raw_events)

    if df_ood.empty:
        print("  Skipped: no OOD customers.")
        return {"status": "skipped", "reason": "no OOD rows"}

    X_ood = df_ood[FEATURE_COLS].astype(float)
    y_ood = df_ood[TARGET_COL].astype(int).to_numpy()
    scores = model.predict_proba(X_ood)[:, 1]
    pred = (scores >= threshold).astype(int)

    actual_rate = float(y_ood.mean())
    result = {
        "status": "ok",
        "n": int(len(df_ood)),
        "actual_churn_rate": actual_rate,
        "synthetic_test_churn_rate": float(synthetic_churn_rate),
        "churn_rate_delta": actual_rate - float(synthetic_churn_rate),
        "mean_predicted_score": float(scores.mean()),
        "selection_rate_at_threshold": float(pred.mean()),
        "recall": float(recall_score(y_ood, pred, zero_division=0)),
        "precision": float(precision_score(y_ood, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_ood, scores)) if len(set(y_ood)) > 1 else None,
    }

    print(f"\n  Raw customers scored:        {result['n']}")
    print(f"  Actual churn rate (raw):     {actual_rate:.4f}")
    print(f"  Synthetic test churn rate:   {synthetic_churn_rate:.4f}   "
          f"(delta {result['churn_rate_delta']:+.4f})")
    print(f"  Mean predicted score:        {result['mean_predicted_score']:.4f}")
    print(f"  Selection rate at threshold: {result['selection_rate_at_threshold']:.4f}")
    if result["roc_auc"] is not None:
        print(f"  ROC-AUC on raw sample:       {result['roc_auc']:.4f}")
    print(f"  Recall {result['recall']:.4f}   Precision {result['precision']:.4f}")

    # A close churn rate says the generator reproduced the label prevalence; a
    # ROC-AUC in the same neighbourhood as the synthetic test says the learned
    # signal transfers. Either one drifting far is the finding.
    if abs(result["churn_rate_delta"]) > 0.10:
        print(f"\n  WARNING: raw churn rate differs from synthetic by "
              f"{abs(result['churn_rate_delta']):.3f} -- the generator's label")
        print("  prevalence does not match the real sample. Revisit the generative constants.")

    return result


def _git_sha() -> str | None:
    """Short git SHA of the working tree, or None outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _file_sha256(path: Path) -> str | None:
    """Content hash of the training input, so a model can be tied to its data."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifacts(
    output_dir: Path,
    model: xgb.XGBClassifier,
    lr_pipeline: Pipeline,
    threshold: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metrics: dict,
    features_path: Path,
    seed: int,
    ood: dict,
) -> dict:
    """Write every artifact the service and the reviewer need. Returns {name: path}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # --- Model, native format -----------------------------------------------
    # save_model, not pickle: the booster JSON is portable across xgboost
    # versions, while a pickled sklearn wrapper breaks the moment the serving
    # container's library version drifts from the training one.
    paths["model"] = output_dir / "churn_model.json"
    model.save_model(paths["model"])

    paths["baseline_lr"] = output_dir / "baseline_lr.joblib"
    joblib.dump(lr_pipeline, paths["baseline_lr"])

    # --- Feature spec --------------------------------------------------------
    recency = X_train["recency_days"]
    sentinel_share = float((recency == NO_SESSION_RECENCY).mean())
    real_recency = recency[recency != NO_SESSION_RECENCY]
    feature_spec = {
        "feature_cols": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "dtypes": {col: str(X_train[col].dtype) for col in FEATURE_COLS},
        "column_order_is_contractual": (
            "The service MUST assemble its matrix in this exact order. XGBoost indexes "
            "features positionally; a reordered matrix scores silently and wrongly."
        ),
        "cutoff": CUTOFF.isoformat(),
        "as_of": AS_OF.isoformat(),
        "cutoff_meaning": (
            "All training features are anchored at CUTOFF: recency is days since the last "
            "session BEFORE CUTOFF, and the 30d/90d windows end there. The label comes only "
            "from [CUTOFF, AS_OF), so no post-cutoff activity can reach a feature."
        ),
        "no_session_recency_sentinel": NO_SESSION_RECENCY,
        "sentinel_meaning": (
            "Customers with no session before CUTOFF get this literal value, not a computed "
            "distance. XGBoost splits it into its own regime; linear models must handle it "
            "separately (see LR_NO_SESSION_COL)."
        ),
        "training_recency_distribution": {
            "mean": float(recency.mean()),
            "p50": float(recency.quantile(0.50)),
            "p90": float(recency.quantile(0.90)),
            "sentinel_share": sentinel_share,
            "mean_excluding_sentinel": float(real_recency.mean()) if len(real_recency) else None,
            "p90_excluding_sentinel": float(real_recency.quantile(0.90)) if len(real_recency) else None,
        },
        "train_serve_skew_note": (
            "EXPECTED DRIFT, not a bug. Training anchors recency at CUTOFF; production scoring "
            "anchors it at the current date, and the gap between now and CUTOFF grows every day "
            "the model stays deployed. Monitor live recency against training_recency_distribution "
            "above and alarm on divergence -- a rising p50 means the feature distribution has "
            "moved out from under the model, and the threshold with it."
        ),
    }
    paths["feature_spec"] = output_dir / "feature_spec.json"
    with paths["feature_spec"].open("w") as fh:
        json.dump(feature_spec, fh, indent=2)

    # --- Threshold -----------------------------------------------------------
    xgb_block = metrics.get("xgboost", {})
    threshold_spec = {
        "threshold": float(threshold),
        "policy": (
            "top-K rank selection for batch; threshold as fallback for single-customer scoring"
        ),
        "target_selection_rate": TARGET_SELECTION_RATE,
        "training_realized_rate": xgb_block.get("val_selection_rate"),
        "test_realized_rate": xgb_block.get("selection_rate"),
        "tie_structure_note": (
            f"{xgb_block.get('best_iteration', 0) + 1} shallow trees produce clustered scores; "
            ">= cut systematically overshoots budget by 5-16%; rank selection guarantees budget "
            "exactly"
        ),
        "monitoring_alert": "raise alert if realized batch selection rate drifts >5% from target",
        "batch_scoring": (
            "Rank customers by score and take the top K = round(target_selection_rate * N). "
            "This guarantees the campaign budget exactly, which a probability cut-off cannot."
        ),
        "single_customer_scoring": (
            "No batch to rank against, so compare the score to `threshold` directly and return "
            "both the score and the comparison."
        ),
    }
    paths["threshold"] = output_dir / "threshold.json"
    with paths["threshold"].open("w") as fh:
        json.dump(threshold_spec, fh, indent=2)

    # --- Metrics -------------------------------------------------------------
    paths["metrics"] = output_dir / "metrics.json"
    with paths["metrics"].open("w") as fh:
        json.dump({**metrics, "out_of_distribution_check": ood}, fh, indent=2)

    # --- Model card ----------------------------------------------------------
    churn_rate = float(y_train.mean())
    model_card = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "random_seed": seed,
        "training_input": str(features_path),
        "training_input_sha256": _file_sha256(features_path),
        "n_train_rows": int(len(X_train)),
        "class_balance": {
            "churned": churn_rate,
            "retained": 1.0 - churn_rate,
            "majority_class": "churned" if churn_rate >= 0.5 else "retained",
            "note": (
                "Churn is the MAJORITY class. Accuracy has a floor equal to the majority rate, "
                "and scale_pos_weight is pinned at 1.0 -- the usual neg/pos upweighting would "
                "push an already-dominant class harder and distort the probabilities the "
                "threshold policy reads."
            ),
        },
        "features": FEATURE_COLS,
        "segments_excluded_from_features": SEGMENT_COLS,
        "synthetic_segments_note": (
            "plan_tier, acquisition_channel and region are SYNTHETIC -- invented in "
            "generate_synthetic.py, with churn associations assigned by hand. They are audit "
            "dimensions only and are excluded from FEATURE_COLS. Every fairness finding is a "
            "statement about this pipeline's ability to DETECT a subgroup gap, NOT evidence "
            "about real-world bias."
        ),
        "chouldechova_note": (
            "Calibration and equal FNR cannot simultaneously hold when base rates differ "
            "(Chouldechova 2017). This model is optimised for calibration, so FNR gaps across "
            "segments at a global threshold are base-rate arithmetic, not model bias. Residual "
            "bias is judged on within-group ROC-AUC instead -- see fairness.json."
        ),
        "training_data_note": (
            "Trained on synthetic customers only. The 80 raw customers are held out entirely "
            "and scored as an out-of-distribution check; see out_of_distribution_check in "
            "metrics.json."
        ),
    }
    paths["model_card"] = output_dir / "model_card.json"
    with paths["model_card"].open("w") as fh:
        json.dump(model_card, fh, indent=2)

    paths["fairness"] = output_dir / "fairness.json"  # written by fairness_audit
    return paths


def main() -> None:
    """CLI: feature table -> trained model, baselines, explanations, audit, artifacts."""
    # Declared up front because --n-top-k rebinds it below: the budget is a single
    # policy value read by the baselines, the threshold derivation and the
    # fairness audit alike, so it stays one source of truth rather than being
    # threaded through five signatures.
    global TARGET_SELECTION_RATE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default=DEFAULT_FEATURES_IN, help=f"Default: {DEFAULT_FEATURES_IN}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"Default: {RANDOM_SEED}")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help=f"Default: {TEST_SIZE}")
    parser.add_argument(
        "--n-top-k",
        type=float,
        default=TARGET_SELECTION_RATE,
        help=f"Campaign budget as a fraction of the base. Default: {TARGET_SELECTION_RATE}",
    )
    parser.add_argument("--raw-events", default=DEFAULT_RAW_EVENTS, help=f"Default: {DEFAULT_RAW_EVENTS}")
    args = parser.parse_args()

    if not 0.0 < args.n_top_k <= 1.0:
        raise ValueError(f"--n-top-k must be a fraction in (0, 1]; got {args.n_top_k}")

    TARGET_SELECTION_RATE = args.n_top_k

    features_path = Path(args.features)
    output_dir = Path(args.output_dir)

    print("=" * 78)
    print("CHURN MODEL TRAINING")
    print("=" * 78)
    print(f"features={features_path}  output={output_dir}  seed={args.seed}  "
          f"test_size={args.test_size}  budget={TARGET_SELECTION_RATE:.0%}")

    X_train, X_test, y_train, y_test, X_lr_train, X_lr_test, df_ood = load_and_split(
        features_path, test_size=args.test_size, seed=args.seed
    )

    baseline_results, lr_pipeline = train_baselines(
        X_train, X_lr_train, y_train, X_test, X_lr_test, y_test
    )

    model, threshold, xgb_metrics = train_xgboost(
        X_train, X_test, y_train, y_test, seed=args.seed, baseline_results=baseline_results
    )

    compute_shap(model, X_test, output_dir, threshold=threshold)

    segments = pd.read_parquet(features_path)[SEGMENT_COLS]
    fairness_df, noise_floor = fairness_audit(model, threshold, X_test, y_test, segments, output_dir)

    ood = score_ood(model, threshold, df_ood, args.raw_events, float(y_test.mean()))

    metrics = {**baseline_results, **xgb_metrics}
    paths = save_artifacts(
        output_dir, model, lr_pipeline, threshold, X_train, y_train,
        metrics, features_path, args.seed, ood,
    )

    _print_summary(metrics, fairness_df, noise_floor, ood, paths, threshold)


def _print_summary(metrics: dict, fairness_df: pd.DataFrame, noise_floor: float,
                   ood: dict, paths: dict, threshold: float) -> None:
    """One block a reviewer can read without scrolling back through the run."""
    xgb_block = metrics["xgboost"]
    rule = metrics["baselines"]["recency_rule"]
    matched = xgb_block.get("matched_audience_comparison", {})

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(f"\nDATA      {metrics['test_n']} test rows, churn rate {metrics['test_churn_rate']:.4f} "
          f"(accuracy floor {metrics['majority_class_accuracy_floor']:.4f})")
    print(f"MODEL     XGBoost, {xgb_block['best_iteration'] + 1} trees, "
          f"threshold p >= {threshold:.4f}, realised selection {xgb_block['selection_rate']:.3f}")

    print(f"\nPERFORMANCE (test)")
    print(f"  ROC-AUC              {xgb_block['roc_auc']:.4f}   (rule {rule['roc_auc']:.4f})")
    print(f"  PR-AUC retention     {xgb_block['pr_auc_retention']:.4f}   (rule {rule['pr_auc_retention']:.4f})")
    print(f"  PR-AUC churn         {xgb_block['pr_auc_churn']:.4f}")
    print(f"  Recall               {xgb_block['recall']:.4f}   "
          f"({xgb_block['pct_max_recall']:.0%} of max {metrics['max_achievable_recall']:.4f} "
          f"reachable at {TARGET_SELECTION_RATE:.0%} budget)")
    print(f"  Precision            {xgb_block['precision']:.4f}")
    print(f"  Brier                {xgb_block['brier']:.4f}   (calibration)")
    print(f"  Confusion            TN {xgb_block['confusion']['tn']}  FP {xgb_block['confusion']['fp']}  "
          f"FN {xgb_block['confusion']['fn']}  TP {xgb_block['confusion']['tp']}")

    if matched:
        delta = matched.get("churners_found_delta", 0)
        print(f"\nVERDICT   At equal {TARGET_SELECTION_RATE:.0%} budget XGBoost finds "
              f"{abs(delta)} {'more' if delta >= 0 else 'fewer'} churners than the recency rule "
              f"({matched['xgboost']['true_positives']} vs "
              f"{matched['recency_rule']['true_positives']}),")
        print(f"          on ROC-AUC {xgb_block['roc_auc'] - rule['roc_auc']:+.4f} better ranking.")

    print(f"\nFAIRNESS  noise floor {noise_floor:.4f} (region control, FNR spread)")
    for segment_col in fairness_df["segment"].unique():
        if segment_col == FAIRNESS_CONTROL_COL:
            continue
        block = fairness_df[fairness_df["segment"] == segment_col]
        control = fairness_df[fairness_df["segment"] == FAIRNESS_CONTROL_COL]
        control_auc = float(control["auc_within_group"].max() - control["auc_within_group"].min())
        verdict, auc_spread, multiple = _bias_verdict(block, control_auc)
        print(f"  {segment_col:<22} AUC spread {auc_spread:.4f} ({multiple:.1f}x control) -> {verdict}")
    print("  Segments are SYNTHETIC -- findings describe detection capability, not real-world bias.")

    if ood.get("status") == "ok":
        print(f"\nOOD CHECK {ood['n']} raw customers: churn {ood['actual_churn_rate']:.4f} vs "
              f"synthetic {ood['synthetic_test_churn_rate']:.4f} "
              f"({ood['churn_rate_delta']:+.4f})", end="")
        print(f", ROC-AUC {ood['roc_auc']:.4f}" if ood.get("roc_auc") is not None else "")

    print("\nARTIFACTS")
    for name, path in sorted(paths.items()):
        print(f"  {name:<14} {path}")


if __name__ == "__main__":
    main()
