# CLAUDE.md - Churn Prediction Service
## Project Purpose
FDE Assignment. Build production ready churn prediction service from raw mobile marketing events.

## Stack
- Python 3.12.3
- Infrastructure: Terraform
- Dashboard: ReatJS or Streamlit

## Key Design Decisions
- Temporal split: feature window BEFORE 2024-05-01, label window AFTER
- Churn label: zero sessions in label window = churned
- No temporal leakage: label derived only from post-cutoff data
- Synthetic data: 8000 customers generated to preserve real sample statistical structure (event mix, timing, amounts)
- Model XGBoost + SHAP explainability, with a Logistic Regression (or simple heuristic) baseline to beat. Evaluate with churn-appropriate metrics (class imbalance-aware; weigh false-negative vs false-positive cost — not raw accuracy). Fairness: check FNR across `plan_tier`, `acquisition_channel`, `region`. Model artifacts → `models/` (gitignored, destined for S3).

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
