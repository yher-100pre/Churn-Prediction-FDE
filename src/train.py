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

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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
