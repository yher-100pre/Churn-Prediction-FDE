# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

FDE take-home. Build a **production-ready churn prediction service** from a raw mobile-marketing event stream (Localytics-style). The deliverable is a shipped raw→RFM→scoring pipeline plus service and infrastructure — not a notebook experiment. Full requirements and grading rubric are in `README.md`; dataset contract is in `data/dataset_schema.json` and `data/README.md`.

## Repository State

This is an early skeleton. Only `src/feature_engineering.py` has real code (temporal constants + `parse_ts`). These are **empty stubs** to be built out: `src/generate_synthetic.py`, `src/train.py`, and everything under `infra/`, `tests/`, `dashboard/`, `models/`, `docs/`. There is no dependency manifest (`requirements.txt`/`pyproject.toml`), test config, or task runner yet — add one when introducing dependencies or tests rather than assuming it exists.

## The Temporal Contract (leakage-critical)

The single most important invariant. Two disjoint windows split by `CUTOFF`, defined in `src/feature_engineering.py`:

```
... feature window ... | CUTOFF | ... label window ... | AS_OF
  (events model sees)  2024-05-01  (defines label)   2024-06-01
```

- **`CUTOFF` = 2024-05-01T12:00:00Z** — feature/label boundary **and the anchor for ALL feature math** (recency + 30d/90d lookbacks). Features use only events strictly before `CUTOFF`.
- **`AS_OF` = 2024-06-01T12:00:00Z** — end of the label observation window **only**. Never a feature/recency anchor.
- **`WINDOW_90D` / `WINDOW_30D`** are derived as `CUTOFF - timedelta(days=...)`, so they inherit CUTOFF's 12:00:00 time-of-day (not midnight) — this keeps them provably consistent with CUTOFF.
- **Recency** = days since last `session` before `CUTOFF`, measured from `CUTOFF`.
- **Label** = zero sessions in `[CUTOFF, AS_OF]` → churned (1), else retained (0).

Why not anchor features at `AS_OF` (which the schema suggests)? That's correct for *production scoring* (score against all history to now), but on this *historical training* dataset it leaks post-CUTOFF activity into features that also define the label. Keep training features anchored at `CUTOFF`. This distinction is documented in the module docstring — preserve it.

## Pipeline Architecture

Intended data flow (build in this order):

1. **`src/generate_synthetic.py`** — the raw sample (`data/events.json`, 800 events / 80 customers) is too small to train on. Generate ~1500 synthetic customers/events preserving the real sample's statistical structure (event-type mix, inter-event timing, session durations, purchase amounts). Also synthesize segment attributes (`plan_tier`, `acquisition_channel`, `region`) — these don't exist in the raw schema and are needed for the fairness check; document them as synthetic. Output → `data/synthetic_events.json` (gitignored).
2. **`src/feature_engineering.py`** — raw events → per-customer RFM table (one row per `customer_id`), carrying the segment columns and the churn label through to `train.py`. Runs against either the raw or synthetic event set (shared schema). Output → `data/training_features.{parquet,csv}` (gitignored).
3. **`src/train.py`** — XGBoost model + SHAP explainability, with a Logistic Regression (or simple heuristic) baseline to beat. Evaluate with churn-appropriate metrics (class imbalance-aware; weigh false-negative vs false-positive cost — not raw accuracy). Fairness: check FNR across `plan_tier`, `acquisition_channel`, `region`. Model artifacts → `models/` (gitignored, destined for S3).

Then: service code exposing scoring, `infra/` Terraform for AWS deployment, and `dashboard/` (React or Streamlit — TBD by time).

## Raw Event Schema

`data/events.json` is a flat JSON array; one object per event: `event_id`, `customer_id`, `event_type`, `timestamp` (ISO-8601 UTC, trailing `Z`), `properties`. Seven event types and their payloads: `session` (`duration_sec`), `purchase` (`amount_usd`), `push_sent`/`push_open` (`campaign_id`), `in_app_event` (`event_name`), `campaign_click` (`campaign_id`), `support_ticket` (`category`: billing|bug|account|other). No profile/demographic fields — hence the synthetic segments above.

## Conventions

- **All timestamps timezone-aware UTC.** Parse event strings with `parse_ts()` (handles trailing `Z`); anchor constants use `tzinfo=timezone.utc`.
- **Generated artifacts stay out of git**: `models/`, synthetic events, feature tables, and Terraform state are gitignored. Models live in S3, not the repo.
- **Reproducibility is graded** — synthetic generation and the RFM transform must be re-runnable; seed and document generative assumptions.

## Stack & Deployment Constraints

- Python 3.12.3; Terraform for infra; React or Streamlit dashboard.
- Preferred AWS building blocks (use only what the architecture needs): S3 (raw + engineered data lake), SageMaker, EKS/EMR/Spark, IAM, Athena, Bedrock, Airflow, Parquet.
- **AWS interview account: 4-day access, shared, strict budget.** Size conservatively (smallest viable EKS node group; prefer batch/serverless scoring over always-on endpoints). Terraform must actually `apply` and the service be reachable — plan-only is insufficient. Capture request/response, logs, and screenshots as evidence since access lapses.
