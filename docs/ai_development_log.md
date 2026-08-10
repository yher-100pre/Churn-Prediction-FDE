# AI Developemtn Log

## Tool: Claude Code
## Engineer: Yenifer Hernandez
## Date: [today]

---

## src/feature_engineering.py

### Approach

### Key Decision — Cutoff vs AS_OF Anchor

Claude Code identified a conflict between two dates in the project:
- CUTOFF = 2024-05-01 (feature/label split boundary from CLAUDE.md)
- AS_OF  = 2024-06-01 (observation date from dataset_schema.json)

Risk: using AS_OF as the recency anchor would allow events from the 
label window (May 1 - Jun 1) to influence feature values temporal 
leakage. A model trained this way looks accurate in training but 
fails in production because it learned from data it would not have 
at prediction time.

Resolution: anchor all features to CUTOFF (May 1). AS_OF marks only 
the end of the label observation window. Documented in module 
docstring with a visual timeline.

In production: when scoring a real customer today, recency should be measured from your prediction date not from a future observation point. Anchoring to CUTOFF mirrors that correctly.

### Key Decision - Helper function to convert timestamp string to a time zone aware utc datetime

To calculate recency both timestamps must follow same time format if not calculations will be off.

**Function build_features function**
All nine features computed from pre_cutoff events only. Label computed first from label_window before any feature calculation no leakage possible by construction.

Notable decisions Claude Code made:
- NO_SESSION_RECENCY = 999 as named module constant — not a magic number: this helps indicate a user never had a previous session and the reasonnwhy NaN or null are not values used is because it can't be processed in the realm of machine learning. And the reason why we don't use 0 is because 0 can mean a session lasted 0 minutes due to a bug or an issue with the platform or app. 
- Shallow copy of event dicts — caller data not mutated
- Half-open interval CUTOFF <= _ts < AS_OF — mathematically precise
- Empty input guard on describe() and churn rate division

Verified: 80 customers, 62.5% churn rate, all features numeric.
Segment columns intentionally excluded — will be merged in from
generate_synthetic.py output per CLAUDE.md design.

**main() and CLI**
Already present in working tree from prior session. Verified 
end to end with .venv/bin/python — important note for README: 
always activate venv before running.

Known limitation noted: missing or malformed input file surfaces 
as raw traceback. Acceptable for now — production version should 
wrap json.load in try/except with a clear error message. 
Will revisit when wired into scheduled pipeline.

Parquet saved with engine="pyarrow" explicitly — required for 
Athena compatibility downstream.

**Design for generate_synthetic.py context verification**

Claude Code identified the critical latent variable design requirement 
independently — if events are generated randomly per customer, RFM 
features carry no information about the label and the model learns 
noise. 

Design confirmed:
- engagement_level ~ Beta(2,2) shifted by segment churn targets
- All behavioral signals (session gaps, purchases, push response) 
  derive from engagement_level
- Churn label emerges from event generation — never stamped directly
- Push funnel is causal: opens only generated as children of sends
- Region gets zero engagement shift — pure fairness control variable
- Sidecar data/synthetic_customers.json keeps event schema pure

Notable insight: Beta(2,2) becomes a tilted distribution within each 
segment tier — expected and correct. Overall churn rate is weighted 
average across tier mix, targeting 60-65%.

**Values generate_synthetic.py constants**
Claude Code measured real sample statistics rather than assuming 
values. Session gaps fit to lognormal(3.409, 1.490) — more accurate 
than exponential for heavy-tailed inter-event data. Purchase tiers 
confirmed from 54 observed purchases. Segment marginals independently 
imply ~65% churn — consistency check passed.

History window set to 365 days before CUTOFF — covers 90d lookback 
with margin, matches observed per-customer active span.

Region deliberately absent from SEGMENT_CHURN_TARGETS — fairness 
control variable. Any FNR gap across region is provably model 
artifact not real signal.

**sample_customer function**
Verified against 1500 draws, seed 42. Marginals match targets 
within sampling noise. Engagement correctly orders by churn 
target across plan_tier and acquisition_channel. Region shows 
unordered noise (0.422-0.454 spread) confirming it is a valid 
fairness control variable.

ALPHA=0.4 produces overlapping engagement distributions across 
segments — no segment is deterministic of label. Engagement 
range compressed to [0.126, 0.753] — expected from blending.

Note for calibration: engagement never reaches 0/1 endpoints 
so behavioral constants are only exercised over middle ~62% 
of their range. Will calibrate purchase:session ratio against 
actual output after Section 3 runs rather than adjusting blind.