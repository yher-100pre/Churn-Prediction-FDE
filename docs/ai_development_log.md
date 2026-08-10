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

**generate_customer_events**
Schema verified: zero orphan funnel children, push_open_rate ≤ 1 
by construction, byte-identical on re-run with seed 42.

Two deliberate deviations from spec documented:
- Push runs on its own clock not per-session — avoids making 
  push_open_rate a proxy for freq_90d
- Support tickets run on their own clock — required to make 
  low-engagement customers file more tickets (friction signal)

Calibration gaps found and resolved:
- PUSH_CADENCE_DAYS corrected to 160 (measured: ~375d/customer 
  in real sample; 160 yields 18.9% push share vs 17.9% target)
- SESSION_GAP parameters retuned: LOG_SD=0.70, SHIFT=3.50
  (LOG_SD reduction defensible: pooled figure conflates 
  within-customer variation with between-customer heterogeneity; 
  engagement now supplies the latter)

Structural finding: ALPHA=0.4 caps segment churn spread at ~+0.17 
vs +0.30 target. Decision: keep ALPHA=0.4 — higher alpha makes 
fairness check measure our own fabrication not model behavior. 
Segment targets restated as ordering priors in documentation.

Bug found and fixed: EVENTS_PER_CUSTOMER upper bound raised from 
45 to 120 — tight upper bound truncates engaged customers mid-
timeline, reads as churn, inverts segment ordering under 
shortened-gap settings.

**build_features update + main()**

Critical finding: engagement_level excluded from sidecar merge 
via explicit SEGMENT_COLUMNS whitelist. Rationale: engagement_level 
is the latent cause of both events AND label — merging it into 
features would be a causal leakage worse than temporal leakage, 
producing perfect synthetic AUC and zero real-world validity.

build_features(events, customers=None) — raw path unchanged: 
still returns 80 customers, 62.5% churn, 11 columns on real sample.
Left join preserves all event customers; sidecar misses print warning 
rather than silently dropping rows.

N_CUSTOMERS raised to 5000: at 1500 the acquisition_channel noise 
floor (0.070) exceeds the signal spread (0.053), scrambling ordering.
At 5000 smallest cell ~290 — ordering resolves correctly.

synthetic_customers.json added to .gitignore.

**Section 1 — train.py constants and imports**
Four decisions documented in module docstring:
- Churn=majority (66%) — accuracy floor 0.64, scale_pos_weight=1.0
- Top-30% budget threshold policy
- Train/serve recency skew — feature_spec.json owes the service 
  the recency anchor and training distribution
- Segments audit-only — detection capability not real-world bias

Three additional constants beyond spec:
- LR_NO_SESSION_COL, LR_RECENCY_CAP — named not inlined
- TARGET_COL = "churned" — single source of truth for string
- FEATURE_COLS order marked contractual with comment

matplotlib added to requirements.txt for SHAP PNG output.

**Logic load_and_split function**
4000 train / 1000 test, stratification tight to 0.0005.
188 customers (3.8%) have no prior session — sentinel handled 
correctly in X_lr, raw 999 preserved in X for XGBoost.

Key design: single index split shared between X and X_lr.
Rationale: two train_test_split calls agree today but 
silently misalign on any argument drift — LR would train 
on shuffled labels and look honestly mediocre, 
indistinguishable from a real gap.

Null routing verified with actual build_features call — 
pandas 3.0 rejects None into bool column, so synthetic 
fixture approach failed. Real data used instead.

OOD check: df_ood correctly empty with synthetic-only parquet.
OOD scoring will call build_features(events.json) separately 
at main() time — raw 80 never touch the split path.

**baselines**
F2 sweep degenerated: F2 is monotonically decreasing in threshold 
when positive class is majority. Picking argmax(F2) collapses to 
trivial baseline (predict everyone churns, threshold=0).

Fix: select all thresholds at TARGET_SELECTION_RATE=0.30 on train.
Makes comparison valid — all models evaluated at same operating point.

Result at 30% selection:
- Recency rule (>106d): recall=0.391, F2=0.436, FN=389
- Logistic Regression: recall=0.391, F2=0.438, FN=389
- Identical recall — recency carries nearly all signal at this budget
- LR wins only on precision (+0.035) and ROC-AUC (0.7221)

max_achievable_recall = 0.30/0.639 = 0.4695
LR achieves 83.3% of maximum achievable — strong performance.
Must be printed alongside recall in all output — without ceiling, 
39% reads as failure.

LR and scaler objects discarded per spec — will refit in 
artifact section if persistence needed.

**baselines verified**
Comparison table complete. Three baselines at consistent 
operating point. Key findings:
- Trivial: sel=1.00, labeled not budget-feasible, shown as floor only
- Recency rule (>106d): sel=0.313, recall=83% of max achievable
- LR: sel=0.300, recall=83% of max achievable, identical FN=389

Protocol fix needed: LR threshold fitted to test scores — 
correcting to train-derived threshold before XGBoost section.
In production, threshold must be frozen from train predictions.
Test selection rate reported as outcome, not target.

Key result for writeup: at 30% budget, recency rule and LR 
produce identical recall. LR edge is precision (+0.035) and 
ranking quality (ROC-AUC 0.722 vs 0.646). XGBoost must answer 
what it buys beyond a one-line recency threshold.

LR Pipeline(scaler + model) returned — joblib round-trip verified.

**Threshold systematic overshoot finding**
Frozen probability threshold overshoots 30% budget in all 15 
seed/derivation combinations tested. Mean realized sel=0.327, 
never below 0.314 against 0.300 target.

Root cause: 46 shallow trees produce clustered scores. 
>= cut at a quantile landing inside a tied block takes 
the whole block — 5-16% campaign budget overrun.

Fix: batch scoring uses top-K rank selection (guarantees 
budget exactly). Frozen threshold retained as:
  1. Monitoring reference — alert if score distribution drifts
  2. Fallback for single-customer real-time scoring where 
     no batch exists to rank against

threshold.json documents both roles explicitly. This changes 
the service design: batch endpoint returns ranked list, 
not binary predictions.