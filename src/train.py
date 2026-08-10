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
            f"Generate it first:  python src/generate_synthetic.py --n-customers 5000\n"
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
