"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts every GTA VI trophy's Exophase earn rate at the case target as five
quantiles, scored by mean pinball loss on the logit scale.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim, no
network, no third-party packages. It reads every `<id>.in` in the input
directory (the grader runs one case at a time) and writes `<id>.out` per case:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

The model, per trophy, in three parts:

1. Trend on the logit scale.  The house reads Exophase on a fixed clock, but
   Exophase refreshes its counts only every few hours, so the history is a
   step function: long flat runs, then a jump.  Consecutive reads that agree
   exactly are collapsed into one "refresh event", timed at the midpoint of
   the gap in which the refresh must have happened.  On those events, the
   logit's trend against x = ln(1 + hours since release) is fitted by
   recency-weighted least squares (player counts grow roughly like a power
   law in time since release, so log-time is the natural axis), anchored at
   the last event so the forecast passes through the last observed rate.
   Slopes are shrunk toward their tier's mean slope (empirical-Bayes, with
   the global mean standing in for tiers with too few trophies), because
   trophies of one tier share a progression shape and single-trophy fits
   over two days are noisy.

2. Earners-over-players check.  rate = E / N, where E is the trophy's
   earner count (rate x players tracked / 100) and N the players tracked,
   so a falling rate under player growth is dilution, not players
   "un-earning" the trophy -- and a rate that holds flat across refreshes
   while N grows means earners are keeping pace, which the trend above
   already reads.  (An earlier draft also extrapolated log E and log N
   separately and averaged the two medians, but that second median is a
   multiplicative move on the rate, which overstates the rise of common
   trophies on the logit scale; the direct logit trend is the honest
   form, so the entry uses it alone.)

3. Spread.  The logit spread combines the fit's residual scatter, the
   shrunk slope's standard error times the extrapolation distance, a
   random-walk term growing with the horizon, and a term for how stale the
   last refresh is.  It is floored (a week of launch dynamics is never
   precisely known) and capped.

Two regimes the trend fit cannot see get their own treatment:

* A flat history (a single distinct snapshot, as in the practice cases):
  the page has not refreshed since the first read, so the last read is
  stale and the true player base has grown unseen.  Launch night the
  tracked players roughly doubled every two hours, so the unseen growth
  to the target is modelled as G = 2 ** ((staleness + horizon) / 2h),
  capped at 12x, and each trophy's rate is diluted by it -- except the
  share new players earn at once, q = (rate/100)**3, which is ~0 for rare
  trophies and 1 for a trophy everyone holds on arrival.  The spread is
  set wide (about 1.2 logit units): a refresh of unknown size is coming.

* A trophy at ~100% (everyone holds it on arrival) stays at ~100% with
  a narrow spread, and a trophy nobody has earned yet gets a small
  positive median (0.05%) with a wide spread.

Everything is deterministic: no randomness, no wall-clock reads, fixed
constants, trophies processed in id order.  Two runs on one case write
byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.  (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# --- model constants (fixed in advance, never tuned on the practice) ---
PRIOR_SLOPE_SD = 0.6    # logit per unit of ln(1 + hours); spread of slopes
SLOPE_CAP = 3.0         # |shrunk slope| is capped here
MOVE_CAP = 3.5          # |median logit move from the last read| is capped here
RECENCY_DAYS = 1.5      # weight = exp(-age_days / RECENCY_DAYS)
RESID_FLOOR = 0.20      # floor on a fit's residual sd (logit units)
RESID_DEFAULT = 0.35    # residual sd when a trophy has no fittable series
RW_SIGMA = 0.30         # random-walk spread, logit units per sqrt(day)
STALE_SIGMA = 0.20      # extra spread per sqrt(day) since the last refresh
SIGMA_FLOOR = 0.50
SIGMA_CAP = 2.0
FLAT_DOUBLING_HOURS = 2.0   # tracked players doubled ~every 2h on launch night
FLAT_GROWTH_CAP = 12.0      # cap on the unseen player-growth factor
FLAT_SIGMA0 = 1.15          # spread when the history is a single snapshot
FLAT_SIGMA_RW = 0.30
NEVER_EARNED_MEDIAN = 0.05  # percent, for a trophy with no earners yet
NEVER_EARNED_SIGMA = 1.5
NEAR_FULL_RATE = 99.5       # a trophy at/above this stays at ~100%
NEAR_FULL_SIGMA = 0.2


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(t: datetime, release: datetime) -> float:
    """ln(1 + hours since release), floored at 0 before release."""
    hours = (t - release).total_seconds() / 3600.0
    return math.log1p(max(hours, 0.0))


def wls_slope(xs, ys, ws):
    """Weighted least squares of y on x.  Returns (slope, se, resid_sd)
    or None when the x values do not vary."""
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    resid = [y - (ybar + slope * (x - xbar)) for x, y in zip(xs, ys)]
    if n > 2:
        var = sum(w * r * r for w, r in zip(ws, resid)) / sw * n / (n - 2)
    else:
        var = 0.0
    resid_sd = math.sqrt(max(var, 0.0))
    # Few points fit exactly; do not let that claim zero uncertainty.
    if n < 4:
        resid_sd = max(resid_sd, RESID_FLOOR)
    se = resid_sd / math.sqrt(sxx)
    return slope, se, resid_sd


def shrink_slope(slope, se, prior_mean):
    """Empirical-Bayes shrinkage of a fitted slope toward a prior mean.
    Returns (shrunk slope, its standard error)."""
    if slope is None or se is None or not math.isfinite(se) or se <= 0:
        return prior_mean, PRIOR_SLOPE_SD
    w_data = 1.0 / (se * se)
    w_prior = 1.0 / (PRIOR_SLOPE_SD * PRIOR_SLOPE_SD)
    shrunk = (slope * w_data + prior_mean * w_prior) / (w_data + w_prior)
    return shrunk, math.sqrt(1.0 / (w_data + w_prior))


def refresh_events(case, release):
    """Collapse the history into refresh events: consecutive reads that
    agree exactly are one snapshot of the page.  An event's time is the
    midpoint of the gap in which the refresh must have happened (the first
    event is timed at its first read).  Returns [(time, players, rates)]."""
    events = []
    prev_key = None
    prev_slot = None
    for read in case["history"]:
        rates = tuple(read["rates"])
        key = (read["players_tracked"], rates)
        slot = parse_time(read["slot"])
        if key != prev_key:
            if events:
                at = prev_slot + (slot - prev_slot) / 2
            else:
                at = slot
            events.append((at, read["players_tracked"], list(read["rates"])))
            prev_key = key
        prev_slot = slot
    return events


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    release = datetime.fromisoformat(case["game"]["release_date"]).replace(
        tzinfo=timezone.utc)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    x_target = log_time(target, release)

    events = refresh_events(case, release)
    # Fits use only post-release events when there are enough of them: the
    # pre-release reads describe reviewers, not the launch population.
    fit_events = [e for e in events if e[0] >= release]
    if len(fit_events) < 3:
        fit_events = events
    last_time, last_n, last_rates = events[-1]
    x_last = log_time(last_time, release)
    dx = x_target - x_last
    staleness_days = max(
        (checkpoint - last_time).total_seconds() / 86400.0, 0.0)

    # Recency weights over the fitting events.
    weights = [math.exp(-max((checkpoint - t).total_seconds() / 86400.0, 0.0)
                       / RECENCY_DAYS) for t, _, _ in fit_events]
    xs = [log_time(t, release) for t, _, _ in fit_events]

    trophies = case["trophies"]
    n_tr = len(trophies)

    # --- first pass: per-trophy fitted slopes for the logit-rate trend
    rate_fits = []   # (slope, se, resid_sd) or None, per trophy
    for i in range(n_tr):
        ys, ys_w, xs_i = [], [], []
        for (t, n, rates), x, w in zip(fit_events, xs, weights):
            r = rates[i]
            if r is None:
                continue
            ys.append(logit(r))
            ys_w.append(w)
            xs_i.append(x)
        rate_fits.append(wls_slope(xs_i, ys, ys_w))

    # Tier-mean slopes, for shrinkage: a tier's mean over its fitted
    # trophies (precision-weighted); tiers with too few fall back to the
    # global mean; with no fitted slope anywhere the prior mean is 0.
    def tier_mean(fits, tier):
        num = den = 0.0
        for (tr, f) in zip(trophies, fits):
            if tr["tier"] != tier or f is None:
                continue
            wgt = 1.0 / (f[1] * f[1] + PRIOR_SLOPE_SD ** 2)
            num += wgt * f[0]
            den += wgt
        return (num / den) if den > 0 else None

    def prior_mean(fits, tier):
        tiers = sorted({tr["tier"] for tr in trophies})
        per_tier = {t: tier_mean(fits, t) for t in tiers}
        have = [v for v in per_tier.values() if v is not None]
        glob = sum(have) / len(have) if have else 0.0
        if per_tier.get(tier) is None:
            return glob
        # A tier with fewer than 3 fitted trophies borrows the global mean.
        count = sum(1 for tr, f in zip(trophies, fits)
                    if tr["tier"] == tier and f is not None)
        return per_tier[tier] if count >= 3 else glob

    rate_prior = {tr["tier"]: prior_mean(rate_fits, tr["tier"])
                  for tr in trophies}

    flat = len(events) < 2
    if flat:
        # The page has not refreshed since the first read: the last read is
        # stale by the whole history, and the true player base has grown
        # unseen since.  Launch night it doubled about every two hours.
        first_time = events[0][0]
        stale_h = max((checkpoint - first_time).total_seconds() / 3600.0, 0.0)
        growth = min(FLAT_GROWTH_CAP,
                     2.0 ** ((stale_h + horizon_hours) / FLAT_DOUBLING_HOURS))
        flat_sigma = min(SIGMA_CAP,
                         FLAT_SIGMA0 + FLAT_SIGMA_RW * math.sqrt(horizon_days))

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            # Last read lacked the trophy: fall back to its last known rate.
            for _, _, rates in reversed(events):
                if rates[i] is not None:
                    rate = rates[i]
                    break
        if rate is None:
            rate = 1.0

        if rate >= NEAR_FULL_RATE:
            # Everyone holds it on arrival; it stays at ~100%.
            med = logit(99.99)
            sigma = NEAR_FULL_SIGMA
        elif rate <= 0.0:
            # Nobody has earned it yet: a small positive rate, wide spread.
            med = logit(NEVER_EARNED_MEDIAN)
            sigma = NEVER_EARNED_SIGMA
        elif flat:
            # Dilute by the unseen player growth; new players earn a
            # trophy at once with probability (rate/100)**3 -- ~0 for rare
            # trophies, 1 for one everyone holds.
            q_new = (rate / 100.0) ** 3
            diluted = (rate + q_new * 100.0 * (growth - 1.0)) / growth
            med = logit(diluted)
            sigma = flat_sigma
        else:
            tier = trophy["tier"]
            y_last = logit(rate)
            # A: direct trend of the logit rate, anchored at the last read.
            fit = rate_fits[i]
            if fit is not None:
                b_rate, se_rate = shrink_slope(
                    max(-SLOPE_CAP, min(SLOPE_CAP, fit[0])), fit[1],
                    rate_prior[tier])
                resid = max(fit[2], RESID_FLOOR)
            else:
                b_rate, se_rate = rate_prior[tier], PRIOR_SLOPE_SD
                resid = RESID_DEFAULT
            b_rate = max(-SLOPE_CAP, min(SLOPE_CAP, b_rate))
            med = y_last + b_rate * dx
            # Keep the median within a sane move from the last read.
            med = max(y_last - MOVE_CAP, min(y_last + MOVE_CAP, med))
            sigma = math.sqrt(
                resid ** 2
                + (se_rate * dx) ** 2
                + (RW_SIGMA ** 2) * horizon_days
                + (STALE_SIGMA ** 2) * staleness_days
                + 0.10 ** 2)
            sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

        rows.append({"id": trophy["id"], "q": [
            round4(clamp(inv_logit(med + z * sigma))) for z in Z]})
    return {"trophies": rows}


def persistence_fallback(case: dict) -> dict:
    """The house persistence forecast (median = last rate, random-walk
    spread).  Used only if the model above fails on a case, so a case is
    never left invalid by a crash."""
    if not case["history"]:
        # No reads at all: a neutral wide forecast, still a valid case.
        return {"trophies": [
            {"id": tr["id"], "q": [0.01, 0.1, 1.0, 5.0, 20.0]}
            for tr in case["trophies"]]}
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0)
        rows.append({"id": trophy["id"], "q": [
            round4(clamp(inv_logit(lg + z * spread))) for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = persistence_fallback(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
