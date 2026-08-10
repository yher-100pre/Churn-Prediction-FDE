"""Synthetic event generation: scale the 800-event raw sample into a trainable set.

The raw sample (data/events.json) is 800 events across 80 customers -- median 10
events each, ~4 sessions each. That is enough to learn the event *shapes* but far
too small to train on: any train/test split or subgroup slice lands in single- or
low-double-digit counts, which cannot support a credible baseline comparison or a
fairness check. This module generates N_CUSTOMERS synthetic customers whose event
streams preserve the raw sample's measured statistical structure -- event-type
mix, inter-event timing, session durations, purchase amounts -- and writes them
in the same schema, so feature_engineering.build_features runs unchanged over
either set.

Latent variable design
----------------------
Every customer is generated from ONE hidden scalar, engagement_level ~ Beta(2,2),
tilted by that customer's segment (see below). Near 0 = disengaged, near 1 =
engaged. That single scalar drives every observable behaviour:

    engagement_level
        |-- mean inter-session gap   (low engagement -> long gaps -> high recency)
        |-- session count in the 30d/90d lookbacks
        |-- session duration_sec
        |-- per-session purchase probability
        `-- push open / campaign click probability

The churn label is NEVER written down. Events are generated across the full
timeline through AS_OF using the same engagement-driven gap process, and whether
a customer happens to emit a `session` inside [CUTOFF, AS_OF) falls out of that
process. build_features then derives `churned` from the events exactly as it does
for the raw sample. Features and label are therefore causally linked through one
latent cause rather than independently fabricated -- which is the whole point: a
model trained here learns the same signal it would learn from real data, instead
of learning a label we stamped on by hand.

Synthetic segment attributes (INVENTED -- not observed)
------------------------------------------------------
The raw schema has no profile or demographic fields, so a fairness analysis has
nothing to slice on. We invent three: plan_tier, acquisition_channel, region.
These are NOT present in, and NOT inferable from, the raw sample -- they are
assumptions, and every conclusion drawn from them rests on those assumptions.
Their churn associations (SEGMENT_CHURN_TARGETS) are business-plausible priors,
not measurements:

- plan_tier: paying customers have sunk cost and switching friction, so churn
  falls as tier rises.
- acquisition_channel: acquisition quality varies -- referrals arrive with intent
  and pre-existing trust, paid social arrives cheapest and least committed.
- region: deliberately assigned NO churn modifier. It is pure noise. That makes
  it a control for the fairness audit: any FNR gap the model shows across region
  is model artifact or sampling noise, not signal it legitimately learned.

Segments tilt the engagement_level draw rather than overriding the label, so the
target churn rates emerge from behaviour. The realised Beta within a segment is a
tilted Beta(2,2), not Beta(2,2) itself -- Beta(2,2) is the population prior.
Observed-vs-target churn is reported per segment after generation so the tilt can
be verified.

Sidecar pattern
---------------
Segments are customer attributes, not events, and forcing them into the event
payload would corrupt the schema that feature_engineering and the raw sample
share. So generation emits two files:

    data/synthetic_events.json      raw events, schema-identical to events.json
    data/synthetic_customers.json   customer_id -> segments, is_synthetic flag

build_features takes the customer sidecar as an OPTIONAL argument and merges the
segment columns in as a final join, leaving the raw-sample path untouched.

Reproducibility
---------------
Seeded with RANDOM_SEED. Same seed + same constants -> byte-identical output.
"""

from __future__ import annotations

import json
import random
from datetime import timedelta

# CUTOFF / AS_OF are the temporal contract and are owned by feature_engineering.
# Importing them (rather than redeclaring) means the generator and the feature
# builder can never drift out of sync -- there is exactly one definition.
from feature_engineering import AS_OF, CUTOFF

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED = 42

# 1500 customers: large enough that a 20% test split leaves ~300 rows, and that
# the smallest segment cell (premium x referral x latam) still holds enough
# customers for a fairness slice to mean something.
N_CUSTOMERS = 1500

# --- Generation timeline -----------------------------------------------------
# The raw sample spans 2022-07 to 2024-05, with a median per-customer active span
# of ~305 days. We generate one year of history before CUTOFF: it covers the 90d
# lookback with margin, matches the observed span, and avoids simulating a
# two-year tail that no feature reads.
HISTORY_DAYS = 365
HISTORY_START = CUTOFF - timedelta(days=HISTORY_DAYS)

# Events continue past CUTOFF to AS_OF so the label window fills organically.
LABEL_WINDOW_DAYS = (AS_OF - CUTOFF).days  # 31

# --- Observed event mix (measured from data/events.json, n=800) --------------
# Reference only: push_open and campaign_click are NOT sampled from this mix --
# they are generated conditionally on a prior push_sent (see funnel rates below),
# which is what makes the funnel structurally valid. Kept here so generated
# output can be validated against the source distribution.
OBSERVED_EVENT_MIX = {
    "session": 0.3875,  # 310/800
    "in_app_event": 0.1800,  # 144/800
    "push_sent": 0.1787,  # 143/800
    "push_open": 0.0762,  # 61/800
    "campaign_click": 0.0737,  # 59/800
    "purchase": 0.0675,  # 54/800
    "support_ticket": 0.0362,  # 29/800
}

# --- Session timing (measured) ----------------------------------------------
# Per-customer session-to-session gaps in the raw sample are strongly right-
# skewed (median 34.2d, mean 66.4d, max 436d), so they are modelled lognormally.
# These are the POPULATION parameters; engagement_level shifts the mean per
# customer (see ENGAGEMENT_GAP_SHIFT).
SESSION_GAP_LOG_MEAN = 3.409  # log-days; exp(3.409) = 30.2d median gap
SESSION_GAP_LOG_SD = 1.490

# How far engagement_level moves the log-gap mean. A fully disengaged customer
# (engagement 0) gets +ENGAGEMENT_GAP_SHIFT on the log mean (~4.5x longer gaps);
# a fully engaged one (engagement 1) gets -ENGAGEMENT_GAP_SHIFT (~4.5x shorter).
# This is the primary mechanism by which engagement becomes recency.
ENGAGEMENT_GAP_SHIFT = 1.50

# --- Session duration (measured) --------------------------------------------
# n=310: min 9, max 424, mean 183.3, median 184.0, sd 57.5.
# Median ~= mean so the raw distribution is near-symmetric; lognormal with the
# measured log parameters reproduces both while ruling out negative durations.
SESSION_DURATION_LOG_MEAN = 5.147  # exp(5.147) = 172s
SESSION_DURATION_LOG_SD = 0.406
SESSION_DURATION_MIN = 9
SESSION_DURATION_MAX = 424

# Engaged customers stay in-app longer. Multiplier applied to the sampled
# duration, interpolated across engagement_level in [0, 1].
DURATION_ENGAGEMENT_RANGE = (0.70, 1.35)

# --- Purchase amounts (measured, empirical price tiers) ---------------------
# The 54 observed amounts are not lognormal -- they cluster into four obvious
# app-store price points with a visible gap between each. Sampling a tier then a
# uniform value inside it reproduces the real shape; a fitted lognormal would
# smear across the empty gaps and invent prices that do not exist.
# Weights are the observed tier counts: 28, 11, 9, 6 of 54.
PURCHASE_TIERS = [
    ((0.99, 2.40), 0.52),  # 28/54 -- impulse / single-item
    ((4.30, 11.10), 0.20),  # 11/54 -- small bundle
    ((17.00, 22.50), 0.17),  # 9/54  -- standard pack
    ((43.00, 59.00), 0.11),  # 6/54  -- premium pack
]

# Probability that a given session is followed by a purchase, interpolated
# across engagement_level. Calibrated so the generated purchase:session ratio
# lands near the observed 54:310 = 0.174.
PURCHASE_PROB_RANGE = (0.04, 0.30)

# --- Push / campaign funnel (measured conversion rates) ---------------------
# Observed: 143 push_sent -> 61 push_open (0.427) and 59 campaign_click (0.413).
# Both children are generated ONLY against a real prior push_sent for the same
# customer and campaign_id, so push_open_rate <= 1.0 holds by construction
# rather than by the defensive clamp in feature_engineering.
PUSH_OPEN_RATE_BASE = 0.427
CAMPAIGN_CLICK_RATE_BASE = 0.413

# Responsiveness scales with engagement: disengaged customers still receive push
# (marketing does not know they are gone) but stop opening it. Multiplier on the
# base rates, interpolated across engagement_level.
PUSH_RESPONSE_ENGAGEMENT_RANGE = (0.15, 1.75)

# Pushes are sent on a marketing cadence independent of engagement -- campaigns
# fire at the whole list. Mean days between pushes to a given customer.
PUSH_CADENCE_DAYS = 21.0

# Campaign identifiers observed in the raw sample (camp_01 .. camp_20).
CAMPAIGN_IDS = [f"camp_{i:02d}" for i in range(1, 21)]

# --- Non-session event rates (measured) -------------------------------------
# in_app_event fires inside a session (144 in_app : 310 sessions = 0.46 each).
IN_APP_PER_SESSION = 0.46

# in_app_event names, weighted by observed counts (34/31/30/27/22 of 144).
IN_APP_EVENT_NAMES = [
    ("search", 0.236),
    ("feature_use", 0.215),
    ("screen_view", 0.208),
    ("add_to_cart", 0.188),
    ("share", 0.153),
]

# support_ticket is friction, not engagement -- it arrives on its own clock
# (29 tickets / 80 customers over ~305 active days). Mean tickets per customer
# per year of history.
SUPPORT_TICKETS_PER_YEAR = 0.44

# Ticket categories, weighted by observed counts (10/8/6/5 of 29).
SUPPORT_TICKET_CATEGORIES = [
    ("other", 0.345),
    ("bug", 0.276),
    ("billing", 0.207),
    ("account", 0.172),
]

# --- Synthetic segments (INVENTED -- see module docstring) ------------------
# Marginal distributions are chosen so the blended churn target lands ~65%,
# a plausible rate for a free-heavy consumer mobile app.
SEGMENT_DISTRIBUTIONS = {
    "plan_tier": [("free", 0.55), ("basic", 0.30), ("premium", 0.15)],
    "acquisition_channel": [
        ("organic", 0.30),
        ("paid_social", 0.25),
        ("email", 0.18),
        ("referral", 0.15),
        ("other", 0.12),
    ],
    "region": [("na", 0.40), ("emea", 0.30), ("apac", 0.20), ("latam", 0.10)],
}

# Target churn rate per segment value. These are ASSUMPTIONS, not measurements.
# region is intentionally absent: no churn modifier, so it stays a pure control
# for the fairness audit (see module docstring).
SEGMENT_CHURN_TARGETS = {
    "plan_tier": {"free": 0.72, "basic": 0.62, "premium": 0.42},
    "acquisition_channel": {
        "paid_social": 0.75,
        "other": 0.70,
        "organic": 0.65,
        "email": 0.60,
        "referral": 0.50,
    },
}

# How hard the segment target pulls the engagement draw. 0.0 = segments are pure
# decoration and the model can learn nothing from them; 1.0 = engagement is a
# deterministic function of segment, so the model would just memorise the tier
# and the fairness check would be measuring our own fabrication. 0.4 leaves the
# Beta draw dominant (60% of the weight) while still producing a segment signal
# strong enough to be learnable -- and, importantly, leaves the segments
# overlapping, so an engaged free-tier customer and a disengaged premium one both
# exist in the data.
SEGMENT_SHIFT_ALPHA = 0.4

# --- Output paths ------------------------------------------------------------
DEFAULT_EVENTS_OUT = "data/synthetic_events.json"
DEFAULT_CUSTOMERS_OUT = "data/synthetic_customers.json"

# Synthetic ids continue past the raw sample's cust_00000..cust_00079 so the two
# sets can be concatenated without collision.
CUSTOMER_ID_OFFSET = 1000


def _weighted_choice(rng: random.Random, distribution: list[tuple[str, float]]) -> str:
    """Draw one value from a [(value, weight), ...] distribution."""
    values = [value for value, _ in distribution]
    weights = [weight for _, weight in distribution]
    return rng.choices(values, weights=weights, k=1)[0]


def sample_customer(rng: random.Random, customer_id: str) -> dict:
    """Draw one synthetic customer: segment attributes plus a latent engagement level.

    Segments are drawn first, then used to tilt the engagement draw. The order
    matters: engagement is the single hidden cause of every event this customer
    will emit (see module docstring), so the segment influence has to be baked in
    here -- before any behaviour exists -- rather than applied to the label later.
    That is what makes the target churn rates emerge from behaviour instead of
    being stamped on.

    The tilt is a weighted blend toward the segment's implied retention:

        engagement = (1 - ALPHA) * beta_draw + ALPHA * (1 - churn_target)

    Beta(2,2) is the population prior -- symmetric, centred at 0.5, with thin
    tails so extreme customers are rare. Blending shifts and narrows it, so the
    per-segment realised distribution is a tilted Beta, not Beta(2,2) itself.
    """
    plan_tier = _weighted_choice(rng, SEGMENT_DISTRIBUTIONS["plan_tier"])
    acquisition_channel = _weighted_choice(rng, SEGMENT_DISTRIBUTIONS["acquisition_channel"])
    # Drawn like any other segment, but deliberately not consulted below: region
    # carries no churn modifier, so it stays a pure control for the fairness
    # audit (see module docstring).
    region = _weighted_choice(rng, SEGMENT_DISTRIBUTIONS["region"])

    # Simple average of the two contributing segment targets. Both marginals
    # independently imply ~0.65 population churn, so averaging them does not drag
    # the aggregate off target.
    churn_target = (
        SEGMENT_CHURN_TARGETS["plan_tier"][plan_tier]
        + SEGMENT_CHURN_TARGETS["acquisition_channel"][acquisition_channel]
    ) / 2.0

    beta_draw = rng.betavariate(2.0, 2.0)
    engagement_level = (1.0 - SEGMENT_SHIFT_ALPHA) * beta_draw + SEGMENT_SHIFT_ALPHA * (1.0 - churn_target)

    return {
        "customer_id": customer_id,
        "plan_tier": plan_tier,
        "acquisition_channel": acquisition_channel,
        "region": region,
        "engagement_level": engagement_level,
        # Flags every row as invented, so nothing downstream can mistake these
        # segments for observed data if the synthetic and raw sets are merged.
        "is_synthetic": True,
    }
