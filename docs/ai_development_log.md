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