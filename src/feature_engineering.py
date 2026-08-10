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

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

# Recency sentinel for customers with no session at all before CUTOFF. Kept far
# outside the plausible range so the model can split it off as its own regime
# instead of treating "never seen" as "seen a long time ago".
NO_SESSION_RECENCY = 999.0

# --- Customer sidecar merge --------------------------------------------------
# Columns build_features will accept from the optional customer sidecar. This is
# a WHITELIST, deliberately, rather than "merge whatever the sidecar has".
#
# The synthetic sidecar also carries `engagement_level`, the latent scalar that
# generates every event and therefore the label itself. Merging it in would hand
# the model the answer -- a leak far worse than the temporal one this module is
# built to prevent, and one that would not show up as anything except suspiciously
# good metrics. Segment columns are fairness-audit dimensions and legitimate
# features; the latent cause is neither. Adding a new segment means adding it
# here explicitly, which is the point: the default is exclusion.
SEGMENT_COLUMNS = ("plan_tier", "acquisition_channel", "region", "is_synthetic")


def parse_ts(ts_str: str) -> datetime:
    """Parse an ISO-8601 timestamp (trailing 'Z' allowed) to a UTC-aware datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)


def build_features(events: list[dict], customers: list[dict] | None = None) -> pd.DataFrame:
    """Collapse a raw event stream into a per-customer RFM + engagement table.

    One row per customer_id. The label comes only from the label window
    [CUTOFF, AS_OF); every feature comes only from events strictly before
    CUTOFF, so no post-CUTOFF signal can reach the model (see module docstring).

    `customers` is an optional sidecar of customer-attribute dicts (as emitted by
    generate_synthetic), each with a `customer_id` plus segment fields. When
    supplied, the columns named in SEGMENT_COLUMNS are joined on as a final step,
    giving the fairness audit something to slice on. Only those columns cross the
    join -- see SEGMENT_COLUMNS for why that whitelist is not negotiable.

    Passing None leaves the raw-sample path exactly as it was: the raw events have
    no profile fields, so there is no sidecar to merge and the output is
    features-plus-label only.
    """
    # Group first so each customer's timeline can be bucketed independently.
    by_customer: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        # Parse once per event, carried on a shallow copy so the caller's dicts
        # are left untouched.
        by_customer[event["customer_id"]].append(dict(event, _ts=parse_ts(event["timestamp"])))

    rows: list[dict] = []
    for customer_id, customer_events in by_customer.items():
        # --- Time buckets -----------------------------------------------------
        pre_cutoff = [e for e in customer_events if e["_ts"] < CUTOFF]
        label_window = [e for e in customer_events if CUTOFF <= e["_ts"] < AS_OF]
        w30 = [e for e in pre_cutoff if e["_ts"] >= WINDOW_30D]
        w90 = [e for e in pre_cutoff if e["_ts"] >= WINDOW_90D]

        # --- Label (computed first, from the label window only) ---------------
        # Churn definition: no session at all in the observation window means the
        # customer stopped opening the app, i.e. churned.
        label_sessions = [e for e in label_window if e["event_type"] == "session"]
        churned = 1 if len(label_sessions) == 0 else 0

        # --- Features (pre-CUTOFF events only) --------------------------------
        pre_sessions = [e for e in pre_cutoff if e["event_type"] == "session"]

        # RECENCY: a long silence before the cutoff is the single strongest churn
        # signal -- disengagement almost always shows up as a growing gap first.
        if pre_sessions:
            last_session_ts = max(e["_ts"] for e in pre_sessions)
            recency_days = (CUTOFF - last_session_ts).total_seconds() / 86400.0
        else:
            recency_days = NO_SESSION_RECENCY  # never active pre-cutoff: maximal churn risk

        # FREQUENCY (30d): recent habit strength -- a customer still opening the
        # app last month is far likelier to open it next month.
        freq_30d = sum(1 for e in w30 if e["event_type"] == "session")

        # FREQUENCY (90d): longer-horizon baseline; separates a genuine regular
        # from a one-off spike, and pairs with freq_30d to expose a slowdown.
        freq_90d = sum(1 for e in w90 if e["event_type"] == "session")

        w90_purchases = [e for e in w90 if e["event_type"] == "purchase"]

        # MONETARY (90d): spend is committed value -- paying customers have
        # sunk cost and switching friction, so they churn less than free riders.
        monetary_90d = float(sum(e.get("properties", {}).get("amount_usd", 0.0) for e in w90_purchases))

        # PURCHASE COUNT (90d): repeat buying beats one big order -- a habit of
        # transacting signals ongoing intent, not a single impulse.
        purchase_count_90d = len(w90_purchases)

        # SESSION DEPTH: short sessions mean shallow engagement; average dwell
        # time falling off is an early warning that precedes outright silence.
        avg_session_duration = (
            float(sum(e.get("properties", {}).get("duration_sec", 0.0) for e in pre_sessions)) / len(pre_sessions)
            if pre_sessions
            else 0.0
        )

        push_sent_count = sum(1 for e in pre_cutoff if e["event_type"] == "push_sent")
        push_open_count = sum(1 for e in pre_cutoff if e["event_type"] == "push_open")
        campaign_click_count = sum(1 for e in pre_cutoff if e["event_type"] == "campaign_click")

        # PUSH RESPONSIVENESS: ignoring notifications shows the app has lost
        # mindshare -- reachability decays before usage does.
        push_open_rate = min(push_open_count / push_sent_count, 1.0) if push_sent_count > 0 else 0.0

        # CAMPAIGN RESPONSIVENESS: clicking through to content is stronger
        # intent than merely opening a push; near-zero means marketing can no
        # longer pull the customer back, which is what makes churn stick.
        click_denominator = push_sent_count + campaign_click_count
        campaign_click_rate = campaign_click_count / click_denominator if click_denominator else 0.0

        # FRICTION: support tickets are logged dissatisfaction (billing disputes,
        # bugs) -- unresolved pain is a leading cause of voluntary churn.
        support_ticket_count = sum(1 for e in pre_cutoff if e["event_type"] == "support_ticket")

        rows.append(
            {
                "customer_id": customer_id,
                "recency_days": recency_days,
                "freq_30d": freq_30d,
                "freq_90d": freq_90d,
                "monetary_90d": monetary_90d,
                "purchase_count_90d": purchase_count_90d,
                "avg_session_duration": avg_session_duration,
                "push_open_rate": push_open_rate,
                "campaign_click_rate": campaign_click_rate,
                "support_ticket_count": support_ticket_count,
                "churned": churned,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("build_features: no events supplied -- empty feature table.")
        return df

    if customers:
        segments = pd.DataFrame(customers)
        # Intersect rather than index directly: a sidecar missing an optional
        # column should not take the whole build down, and any column NOT in the
        # whitelist (engagement_level above all) is dropped here.
        keep = ["customer_id"] + [c for c in SEGMENT_COLUMNS if c in segments.columns]
        # Left join: the event stream decides who is in the table. A customer in
        # the sidecar with no events has no features and must not appear as a row
        # of nulls; one with events but no sidecar entry keeps its features and
        # gets null segments, which the fairness slice can then exclude explicitly
        # rather than silently.
        df = df.merge(segments[keep].drop_duplicates("customer_id"), on="customer_id", how="left")

        missing = int(df[keep[1]].isna().sum()) if len(keep) > 1 else 0
        if missing:
            print(f"build_features: {missing} customers had events but no sidecar entry -- segments are null.")

    print(f"Customers: {len(df)}")
    print(f"Churn rate: {df['churned'].mean() * 100:.1f}%")
    print(f"Feature means: {df.describe().loc['mean'].round(2).to_dict()}")

    return df


def main() -> None:
    """CLI: raw event JSON -> per-customer RFM table on disk (parquet + csv)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        default="data/events.json",
        help="Raw event stream JSON (raw sample or synthetic). Default: data/events.json",
    )
    parser.add_argument(
        "--output",
        default="data/rfm_features.parquet",
        help="Parquet output path; the .csv twin is derived from it. Default: data/rfm_features.parquet",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open() as fh:
        events = json.load(fh)
    print(f"Loaded {len(events)} events from {input_path}")

    df = build_features(events)

    parquet_path = Path(args.output)
    csv_path = parquet_path.with_suffix(".csv")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # pyarrow explicitly: Athena/Glue read the parquet straight out of S3, so the
    # writer must be the one whose type mapping they expect (not fastparquet).
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    # CSV twin for eyeballing and quick diffs; parquet stays the canonical artifact.
    df.to_csv(csv_path, index=False)

    print(f"Wrote parquet: {parquet_path}")
    print(f"Wrote csv:     {csv_path}")


if __name__ == "__main__":
    main()
