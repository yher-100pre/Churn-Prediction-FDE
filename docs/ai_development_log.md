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