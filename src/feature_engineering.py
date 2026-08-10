"""Feature engineering: raw event stream -> per-customer RFM training table.

Transforms the raw mobile-marketing event stream (events.json / synthetic
equivalent) into a customer-level feature table (one row per customer_id)
suitable for training and scoring a churn model. Bridges the raw events and
train.py.

Temporal design (no leakage)
----------------------------
Two disjoint time windows separated by CUTOFF:

    ... feature window ...  | CUTOFF |  ... label window ...  | AS_OF
      (events model sees)   2024-05-01   (defines the label)   2024-06-01

- Features are computed ONLY from events strictly before CUTOFF.
- The churn label is derived ONLY from events in [CUTOFF, AS_OF].
- Recency and the 30d/90d lookback windows are anchored at CUTOFF, not AS_OF.

Why CUTOFF (not AS_OF) is the recency anchor
--------------------------------------------
The dataset schema instructs using AS_OF (2024-06-01) as the observation
timestamp for recency/lookback. That is correct in a *production scoring*
context: you score a customer today against all history up to now, and there
is no future to leak from. But in *training* on a historical dataset, anchoring
features at AS_OF would let post-CUTOFF activity bleed into the features while
that same post-CUTOFF activity also defines the label -- circular, and it
inflates apparent accuracy. To keep the training features strictly leak-free,
every feature calculation here is anchored at CUTOFF. AS_OF is used only to
bound the label observation window.

Constants
---------
- AS_OF      2024-06-01T12:00:00Z  end of label window / observation close.
- CUTOFF     2024-05-01T12:00:00Z  feature/label boundary and recency anchor.
- WINDOW_90D 2024-02-01            90 days before CUTOFF (lookback start).
- WINDOW_30D 2024-04-01            30 days before CUTOFF (lookback start).

Recency = days since the last session strictly before CUTOFF, measured from CUTOFF.
Label   = zero sessions in [CUTOFF, AS_OF]  ->  churned (1), else retained (0).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

# --- Temporal anchors (all UTC) ---------------------------------------------
# End of the label observation window: when the observation period closes.
# Used ONLY to bound the label window -- never as a feature/recency anchor.
AS_OF = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

# Feature/label split boundary and the anchor for ALL feature calculations
# (recency + lookback windows). Features use events strictly before CUTOFF.
CUTOFF = datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc)

# Lookback window start points, defined relative to CUTOFF.
WINDOW_90D = CUTOFF - timedelta(days=90)  # 2024-02-01T12:00:00Z
WINDOW_30D = CUTOFF - timedelta(days=30)  # 2024-04-01T12:00:00Z


def parse_ts(ts_str: str) -> datetime:
    """Parse an ISO-8601 timestamp (trailing 'Z' allowed) to a UTC-aware datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)
