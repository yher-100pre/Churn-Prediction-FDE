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

SEGMENT_CHURN_TARGETS are ORDERING PRIORS, not precise targets
--------------------------------------------------------------
The realised per-segment churn rates do NOT reproduce SEGMENT_CHURN_TARGETS, and
are not meant to. The targets fix the ORDER and the SIGN of each segment's effect
-- premium churns less than basic, which churns less than free; referral churns
less than paid_social -- while the realised spread comes out roughly half the
nominal one (~0.13 across plan_tier against a nominal 0.30).

This is a deliberate consequence of SEGMENT_SHIFT_ALPHA = 0.4, not a calibration
failure. Alpha is the dial between two things we cannot have at once:

    low alpha   segments barely move engagement; realised spread is compressed,
                and the segment columns carry weak (but honest) signal.
    high alpha  realised churn matches the targets closely -- because engagement,
                and therefore the label, becomes close to a deterministic
                function of segment.

At high alpha the fairness audit stops being informative. Any FNR gap the model
showed across plan_tier would simply be the model recovering the tilt we wrote
into the generator, so the audit would be measuring our own construction rather
than model behaviour. Because these segments are invented (see above), that
finding would say nothing about the model at all. Holding alpha at 0.4 keeps the
segments overlapping -- engaged free-tier and disengaged premium customers both
exist in quantity -- so a measured FNR gap reflects what the model does with weak
signal, which is the question the audit is actually asking.

Read the per-segment table printed by main() as a direction-and-ordering check.
Treat any exact agreement with the nominal targets as a red flag: it would mean
alpha has drifted high enough to compromise the audit.

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

import argparse
import json
import random
from datetime import timedelta
from pathlib import Path

# CUTOFF / AS_OF are the temporal contract and are owned by feature_engineering.
# Importing them (rather than redeclaring) means the generator and the feature
# builder can never drift out of sync -- there is exactly one definition.
from feature_engineering import AS_OF, CUTOFF, build_features

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED = 42

# 8000 customers: large enough that a 20% test split leaves ~1600 rows, and that
# the smallest segment cell (premium x referral x latam) still holds enough
# customers for a fairness slice to mean something.
N_CUSTOMERS = 8000

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

# --- Session timing (measured, then calibrated) ------------------------------
# Per-customer session-to-session gaps in the raw sample are strongly right-
# skewed (median 34.2d, mean 66.4d, max 436d), so they are modelled lognormally.
# The measured pooled fit is (log_mean 3.409, log_sd 1.490). Both were retuned:
#
# log_sd 1.490 -> 0.700. The pooled sd conflates WITHIN-customer gap variation
#   with BETWEEN-customer heterogeneity. Here engagement_level already supplies
#   the between-customer spread, so keeping the pooled sd double-counts it -- and
#   at 1.490 the per-draw noise is larger than the entire engagement-driven
#   signal range, which flattened churn to within 0.7pp across plan_tier
#   (premium 0.766 vs free 0.773). 0.700 is the within-customer residual.
#
# log_mean 3.409 -> 4.000. Once the sd shrank, the shorter effective gaps drove
#   overall churn well under target; the mean was re-solved by bisection to land
#   the realised churn rate in [0.63, 0.67]. exp(4.000) = 54.6d median gap at
#   engagement 0.5. Re-run main() after any change here -- these two constants
#   and the churn rate move together.
SESSION_GAP_LOG_MEAN = 4.000
SESSION_GAP_LOG_SD = 0.700

# How far engagement_level moves the log-gap mean. A fully disengaged customer
# (engagement 0) gets +ENGAGEMENT_GAP_SHIFT on the log mean; a fully engaged one
# (engagement 1) gets -ENGAGEMENT_GAP_SHIFT. This is the primary mechanism by
# which engagement becomes recency.
#
# Raised 1.50 -> 3.50 alongside the sd cut above. It has to be this large because
# the realised engagement spread between segments is narrow by design: ALPHA=0.4,
# halved again by averaging two segment fields, turns the 0.30-wide plan_tier
# target range into only ~0.07 of engagement separation (free 0.42, premium 0.49).
# A large shift is what converts that thin margin into a visible behavioural
# difference. See the module docstring on ordering priors.
ENGAGEMENT_GAP_SHIFT = 3.50

# Ceiling on session anchors per customer, interpolated across engagement_level.
# This is a SAFETY CEILING, not a second churn mechanism: the gap walk above is
# what decides how many sessions a customer emits, because the walk stops at
# AS_OF. The ceiling only catches a freak short-gap streak from producing absurd
# volume.
#
# It must stay ABOVE what the timeline can hold at high engagement, or it becomes
# actively harmful: a customer who exhausts the budget stops emitting, and a
# customer who stops emitting before CUTOFF is labelled churned. Truncating the
# MOST engaged customers is exactly backwards, and it silently flattens (at a
# tight enough setting, inverts) the churn ordering across segments -- the signal
# the whole generator exists to produce. Measured: at the shortest gaps any
# plausible calibration reaches, an engaged customer needs ~40 sessions to walk
# from HISTORY_START to AS_OF, so the upper bound is set to 3x that. Any retuning
# that shortens SESSION_GAP_LOG_MEAN must re-check this headroom.
EVENTS_PER_CUSTOMER = (4, 120)

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
#
# Corrected 21.0 -> 160.0 against the raw sample. 21 days was an assumption, and
# it was badly wrong: it made push_sent 44.6% of all generated events against an
# observed 17.9%, drowning the session signal in notification traffic. The raw
# sample has 143 sends across 80 customers over a ~670-day span -- roughly one
# push per customer per year. 160d reproduces the observed push share.
PUSH_CADENCE_DAYS = 160.0

# How long after a send its children land, in seconds. A push is opened within
# hours or not at all; a click follows the open within the same sitting.
PUSH_OPEN_DELAY_SEC = (60, 6 * 3600)
CLICK_AFTER_OPEN_DELAY_SEC = (10, 1800)

# Campaign identifiers observed in the raw sample (camp_01 .. camp_20).
CAMPAIGN_IDS = [f"camp_{i:02d}" for i in range(1, 21)]

# --- Non-session event rates (measured) -------------------------------------
# in_app_event fires inside a session (144 in_app : 310 sessions = 0.46 each).
IN_APP_PER_SESSION = 0.46

# Engaged customers do more per visit -- search, add to cart, share. Multiplier
# on IN_APP_PER_SESSION, interpolated across engagement_level. The result is the
# EXPECTED number of children; they are emitted as up to MAX_IN_APP_PER_SESSION
# independent Bernoulli draws, so a session yields 0, 1 or 2.
IN_APP_ENGAGEMENT_RANGE = (0.50, 1.60)
MAX_IN_APP_PER_SESSION = 2

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

# Friction runs slightly AGAINST engagement: a customer fighting a billing
# dispute or a bug is on their way out, so the disengaged file marginally more
# tickets despite visiting less. Multiplier on the yearly rate, interpolated
# across engagement_level -- note it descends, unlike every other range here.
# Kept mild (1.6x -> 0.7x): tickets are a weak signal and overstating them would
# hand the model an artificial shortcut to the label.
SUPPORT_TICKET_ENGAGEMENT_RANGE = (1.60, 0.70)

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

# Target churn rate per segment value. These are ASSUMPTIONS, not measurements,
# and they are ORDERING PRIORS rather than rates the generator will reproduce --
# the realised spread lands at roughly half the nominal one, by design. See
# "SEGMENT_CHURN_TARGETS are ORDERING PRIORS" in the module docstring before
# changing these or SEGMENT_SHIFT_ALPHA.
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
DEFAULT_FEATURES_OUT = "data/training_features.parquet"

# Synthetic ids continue past the raw sample's cust_00000..cust_00079 so the two
# sets can be concatenated without collision.
CUSTOMER_ID_OFFSET = 1000


def _weighted_choice(rng: random.Random, distribution: list[tuple[str, float]]) -> str:
    """Draw one value from a [(value, weight), ...] distribution."""
    values = [value for value, _ in distribution]
    weights = [weight for _, weight in distribution]
    return rng.choices(values, weights=weights, k=1)[0]


def _lerp(bounds: tuple[float, float], t: float) -> float:
    """Interpolate between bounds at position t. t is an engagement_level in [0, 1]."""
    low, high = bounds
    return low + (high - low) * t


def _poisson(rng: random.Random, mean: float) -> int:
    """Draw a Poisson count (Knuth). Used for the low-rate support-ticket clock."""
    if mean <= 0.0:
        return 0
    threshold = 2.718281828459045 ** -mean
    count, product = 0, 1.0
    while True:
        product *= rng.random()
        if product <= threshold:
            return count
        count += 1


def _iso_z(ts) -> str:
    """Format a UTC-aware datetime in the raw sample's exact wire format."""
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


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


def generate_customer_events(rng: random.Random, customer: dict, history_start) -> list[dict]:
    """Expand one customer dict into a raw event stream in the events.json schema.

    Three independent clocks run from history_start to AS_OF:

        session clock   engagement-driven lognormal gaps. Sessions carry their
                        own children (in_app_event, purchase) inside the visit.
        push clock      marketing cadence, INDEPENDENT of engagement -- campaigns
                        fire at the whole list. push_open and campaign_click are
                        emitted only against a real prior send, never on their own.
        ticket clock    Poisson friction, mildly inverse to engagement.

    Why the push and ticket clocks are not nested in the session loop: both would
    then inherit the session count, which is the thing engagement already drives.
    Push volume would become a proxy for visit frequency (making push_open_rate
    partly a restatement of freq_90d rather than an independent signal), and
    tickets would run WITH engagement instead of against it -- a disengaged
    customer would file fewer, inverting the friction signal they are there to
    carry. The funnel rule that matters is preserved exactly: every push_open and
    campaign_click hangs off a specific push_sent and shares its campaign_id.

    The churn label is never decided here. Events simply stop when the walk
    passes AS_OF, and whether a session happens to land in [CUTOFF, AS_OF) is an
    outcome of the gap process. Returns events sorted by timestamp; event_ids are
    assigned last, so they run in chronological order per customer.
    """
    engagement = customer["engagement_level"]
    customer_id = customer["customer_id"]

    # (timestamp, event_type, properties) triples; ids and formatting come later.
    raw: list[tuple] = []

    # --- Session clock --------------------------------------------------------
    # Engagement moves the log-gap mean symmetrically about the population value:
    # engagement 0 -> +SHIFT (long gaps), 0.5 -> unshifted, 1 -> -SHIFT (short).
    gap_log_mean = SESSION_GAP_LOG_MEAN + ENGAGEMENT_GAP_SHIFT * (1.0 - 2.0 * engagement)
    max_sessions = max(1, round(_lerp(EVENTS_PER_CUSTOMER, engagement)))
    duration_multiplier = _lerp(DURATION_ENGAGEMENT_RANGE, engagement)
    purchase_prob = _lerp(PURCHASE_PROB_RANGE, engagement)
    in_app_prob = min(
        IN_APP_PER_SESSION * _lerp(IN_APP_ENGAGEMENT_RANGE, engagement) / MAX_IN_APP_PER_SESSION,
        1.0,
    )

    # Phase-in: the first session lands a random FRACTION of a gap after
    # history_start, so customers do not all begin in lockstep at the window edge.
    ts = history_start + timedelta(days=rng.lognormvariate(gap_log_mean, SESSION_GAP_LOG_SD) * rng.random())

    for _ in range(max_sessions):
        if ts >= AS_OF:
            break

        duration = rng.lognormvariate(SESSION_DURATION_LOG_MEAN, SESSION_DURATION_LOG_SD) * duration_multiplier
        duration = int(round(min(max(duration, SESSION_DURATION_MIN), SESSION_DURATION_MAX)))
        raw.append((ts, "session", {"duration_sec": duration}))

        # Children fire somewhere inside the visit, so they can never outlive it.
        for _ in range(MAX_IN_APP_PER_SESSION):
            if rng.random() < in_app_prob:
                child_ts = ts + timedelta(seconds=rng.uniform(0, duration))
                raw.append((child_ts, "in_app_event", {"event_name": _weighted_choice(rng, IN_APP_EVENT_NAMES)}))

        if rng.random() < purchase_prob:
            bounds = rng.choices(
                [b for b, _ in PURCHASE_TIERS], weights=[w for _, w in PURCHASE_TIERS], k=1
            )[0]
            purchase_ts = ts + timedelta(seconds=rng.uniform(0, duration))
            raw.append((purchase_ts, "purchase", {"amount_usd": round(rng.uniform(*bounds), 2)}))

        ts += timedelta(days=rng.lognormvariate(gap_log_mean, SESSION_GAP_LOG_SD))

    # --- Push clock -----------------------------------------------------------
    # Sends are exponential on the marketing cadence and ignore engagement;
    # only the RESPONSE to them scales, which is what makes push_open_rate decay
    # for the disengaged while push_sent_count stays flat across the population.
    response_multiplier = _lerp(PUSH_RESPONSE_ENGAGEMENT_RANGE, engagement)
    open_prob = min(PUSH_OPEN_RATE_BASE * response_multiplier, 1.0)
    click_prob = min(CAMPAIGN_CLICK_RATE_BASE * response_multiplier, 1.0)

    push_ts = history_start + timedelta(days=rng.expovariate(1.0 / PUSH_CADENCE_DAYS) * rng.random())
    while push_ts < AS_OF:
        campaign_id = rng.choice(CAMPAIGN_IDS)
        raw.append((push_ts, "push_sent", {"campaign_id": campaign_id}))

        open_ts = None
        if rng.random() < open_prob:
            candidate = push_ts + timedelta(seconds=rng.uniform(*PUSH_OPEN_DELAY_SEC))
            if candidate < AS_OF:
                open_ts = candidate
                raw.append((open_ts, "push_open", {"campaign_id": campaign_id}))

        if rng.random() < click_prob:
            # A click through an opened push follows it in the same sitting;
            # a click without an open (deep link, in-feed ad) uses the send.
            if open_ts is not None:
                click_ts = open_ts + timedelta(seconds=rng.uniform(*CLICK_AFTER_OPEN_DELAY_SEC))
            else:
                click_ts = push_ts + timedelta(seconds=rng.uniform(*PUSH_OPEN_DELAY_SEC))
            if click_ts < AS_OF:
                raw.append((click_ts, "campaign_click", {"campaign_id": campaign_id}))

        push_ts += timedelta(days=rng.expovariate(1.0 / PUSH_CADENCE_DAYS))

    # --- Support-ticket clock -------------------------------------------------
    span_days = (AS_OF - history_start).total_seconds() / 86400.0
    ticket_mean = (
        SUPPORT_TICKETS_PER_YEAR * _lerp(SUPPORT_TICKET_ENGAGEMENT_RANGE, engagement) * span_days / 365.0
    )
    for _ in range(_poisson(rng, ticket_mean)):
        ticket_ts = history_start + timedelta(days=rng.uniform(0, span_days))
        raw.append((ticket_ts, "support_ticket", {"category": _weighted_choice(rng, SUPPORT_TICKET_CATEGORIES)}))

    # --- Serialise ------------------------------------------------------------
    # Chronological ids make a customer's stream readable top-to-bottom, and
    # sorting cannot break the funnel: every child was built after its parent.
    raw.sort(key=lambda item: item[0])
    return [
        {
            "event_id": f"syn_{customer_id}_{index:06d}",
            "customer_id": customer_id,
            "event_type": event_type,
            "timestamp": _iso_z(event_ts),
            "properties": properties,
        }
        for index, (event_ts, event_type, properties) in enumerate(raw)
    ]


def generate_dataset(rng: random.Random, n_customers: int) -> tuple[list[dict], list[dict]]:
    """Draw n_customers and expand each into events. Returns (customers, events).

    Events come back sorted globally by timestamp, matching the raw sample's
    layout. That is safe for the funnel because event_ids were already assigned
    per customer in chronological order, so a global sort reorders rows without
    touching any parent/child relationship.
    """
    customers = [sample_customer(rng, f"cust_{CUSTOMER_ID_OFFSET + i:05d}") for i in range(n_customers)]

    events: list[dict] = []
    for customer in customers:
        events.extend(generate_customer_events(rng, customer, HISTORY_START))

    events.sort(key=lambda event: event["timestamp"])
    return customers, events


def _print_calibration(events: list[dict], features) -> None:
    """Report the generated set against the raw sample and the segment priors.

    Read the per-segment block as a direction-and-ordering check, not a hit-the-
    number check -- see the module docstring on ordering priors.
    """
    counts: dict[str, int] = {}
    for event in events:
        counts[event["event_type"]] = counts.get(event["event_type"], 0) + 1
    total = len(events)

    print("\n" + "=" * 66)
    print("CALIBRATION")
    print("=" * 66)
    print(f"Customers {len(features)}   Events {total}   Events/customer {total / len(features):.1f}")

    print("\nEvent mix                generated       raw sample      delta")
    for event_type, observed_share in OBSERVED_EVENT_MIX.items():
        share = counts.get(event_type, 0) / total
        print(f"  {event_type:<20} {share:>9.4f} {observed_share:>15.4f} {share - observed_share:>+11.4f}")

    push_share = counts.get("push_sent", 0) / total
    ratio = counts.get("purchase", 0) / max(counts.get("session", 0), 1)
    print(f"\npurchase:session ratio     {ratio:>9.3f} {0.174:>15.3f} {ratio - 0.174:>+11.3f}")
    print(f"push_sent share            {push_share:>9.4f} {OBSERVED_EVENT_MIX['push_sent']:>15.4f} "
          f"{push_share - OBSERVED_EVENT_MIX['push_sent']:>+11.4f}")

    churn_rate = features["churned"].mean()
    print(f"\nOverall churn rate         {churn_rate:>9.3f} {'0.630-0.670':>15}   "
          f"{'IN BAND' if 0.63 <= churn_rate <= 0.67 else 'OUT OF BAND'}")

    for field in ("plan_tier", "acquisition_channel", "region"):
        if field not in features.columns:
            continue
        targets = SEGMENT_CHURN_TARGETS.get(field, {})
        print(f"\nChurn by {field}:")
        grouped = features.groupby(field)["churned"].agg(["mean", "count"]).sort_values("mean")
        for value, row in grouped.iterrows():
            target = targets.get(value)
            suffix = f"prior {target:.2f}" if target is not None else "no prior -- fairness control"
            print(f"  {value:<14} n={int(row['count']):<5} churn={row['mean']:.3f}   {suffix}")


def main() -> None:
    """CLI: generate the synthetic set, build features, write all three artifacts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-customers", type=int, default=N_CUSTOMERS, help=f"Default: {N_CUSTOMERS}")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"Default: {RANDOM_SEED}")
    parser.add_argument("--output-events", default=DEFAULT_EVENTS_OUT, help=f"Default: {DEFAULT_EVENTS_OUT}")
    parser.add_argument("--output-customers", default=DEFAULT_CUSTOMERS_OUT, help=f"Default: {DEFAULT_CUSTOMERS_OUT}")
    parser.add_argument("--output-features", default=DEFAULT_FEATURES_OUT, help=f"Default: {DEFAULT_FEATURES_OUT}")
    args = parser.parse_args()

    print(f"Generating {args.n_customers} customers (seed {args.seed}) "
          f"over {HISTORY_DAYS}d history + {LABEL_WINDOW_DAYS}d label window...")
    rng = random.Random(args.seed)
    customers, events = generate_dataset(rng, args.n_customers)
    print(f"Generated {len(events)} events.")

    events_path = Path(args.output_events)
    customers_path = Path(args.output_customers)
    features_path = Path(args.output_features)
    csv_path = features_path.with_suffix(".csv")
    for path in (events_path, customers_path, features_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with events_path.open("w") as fh:
        json.dump(events, fh, indent=1)

    # The sidecar keeps engagement_level: it is the generative ground truth and
    # the only way to audit the tilt after the fact. It is NOT a feature -- the
    # SEGMENT_COLUMNS whitelist in feature_engineering is what keeps it out of the
    # training table, since it causes the label outright.
    with customers_path.open("w") as fh:
        json.dump(customers, fh, indent=1)

    print("\n--- build_features ---")
    features = build_features(events, customers)

    features.to_parquet(features_path, engine="pyarrow", index=False)
    features.to_csv(csv_path, index=False)

    _print_calibration(events, features)

    print(f"\nWrote events:   {events_path}  ({len(events)} events)")
    print(f"Wrote sidecar:  {customers_path}  ({len(customers)} customers)")
    print(f"Wrote features: {features_path}")
    print(f"Wrote features: {csv_path}")


if __name__ == "__main__":
    main()
