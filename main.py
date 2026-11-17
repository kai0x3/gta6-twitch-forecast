"""gta6-twitch-ccu-forecast-01 entry: an hour-by-hour Twitch viewers forecast.

The idea: a launch day's shape, measured from its first hour, repeats across
big launches. For every past launch in the case's `comparables` whose release
day is fully recorded, take the log-ratio of each hour's viewers to that
launch's own hour-1 viewers. Pool those ratios hour by hour, weighting each
comparable by how much it resembles this case (a rolling midnight-local
console unlock like this one, no early-access head start, a hot start, a
similar hour-1 size, a similar pre-midnight ramp), blend the weighted pool
with the plain pool, and read the five scored quantiles off a lognormal fit
to the pooled ratios. The forecast for hour h is then

    q_tau(h) = p1 * exp(quantile_tau of the pooled log-ratios at hour h)

with p1 this case's hour-1 print and the pooled median lightly smoothed
across neighbouring hours. The weighting and spread constants were fitted
by leave-one-out backtests over the comparables themselves. Everything is
computed from the case file alone: no network, no randomness, and the same
bytes on every run.

Run it the way the grader does (Python 3.13, standard library only):
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader runs python3 -P, which keeps this file's directory off
# sys.path; nothing else is imported, but keep the path sane anyway.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS, for the persistence fallback.
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
HOURS = list(range(2, 25))

# --- model constants (fitted by leave-one-out backtests on the comparables)
W_UNLOCK = {"rolling": 1.0, "unknown": 0.55, "global": 0.22}
W_EARLY_ACCESS_DIRTY = 0.45   # early access that visibly flattened the day
W_EARLY_ACCESS = 0.7          # any other early access mention
W_F2P = 0.6                   # free-to-play launches behave differently
MAG_SIGMA = 0.85              # log-space width of the size-similarity weight
RAMP_SIGMA = 1.0              # log-space width of the pre-midnight ramp weight
HOT_START_FLOOR = 0.25        # weight floor for a cold-start comparable
HOT_START_FULL = 0.30         # hour-1/peak ratio at which the weight is full
POOL_BLEND = 0.85             # weighted pool vs unweighted pool, in log space
PARAM_BLEND = 1.0             # lognormal tails vs empirical pooled quantiles
SPREAD_CAL = 1.0              # multiplies the pooled log-spread
RATIO_CAP = 3.5               # pooled log-ratios are clamped to +/- this
MIN_POOL = 4                  # usable curves below this -> persistence fallback


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp ('...Z') to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comp):
    """A comparable's release-day viewers by hour 1..24 (missing = absent)."""
    try:
        start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    by_time = {}
    for row in comp.get("hourly") or []:
        try:
            by_time[parse_ts(row["from"])] = row.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
    curve = {}
    for hour in range(1, 25):
        value = by_time.get(start + timedelta(hours=hour - 1))
        if isinstance(value, (int, float)) and value == value:  # not NaN
            curve[hour] = float(value)
    return curve


def is_launch(comp):
    """False for the one comparable that is not a launch at all: the target
    game's own category recorded before release (a premiere, not a day 1)."""
    name = (comp.get("name") or "").lower()
    unlock = (comp.get("unlock") or "").lower()
    return "before release" not in name and "not a launch" not in unlock


def unlock_class(comp):
    """Classify a comparable's unlock pattern from its research note."""
    text = (comp.get("unlock") or "").lower()
    if "midnight local" in text or "midnight, local" in text:
        return "rolling"
    if "global" in text or "simultaneous" in text or "single" in text:
        return "global"
    return "unknown"


def pre_level(rows, day_start):
    """Mean viewers over the six hours before a day starts, from hourly
    rows (a comparable's `hourly`, or a case's `history_hourly`); None
    when the window holds no usable row."""
    if not rows:
        return None
    lo = day_start - timedelta(hours=6)
    values = []
    for row in rows:
        try:
            at = parse_ts(row.get("from") or row.get("at"))
            viewers = row.get("viewers")
        except (TypeError, ValueError):
            continue
        if lo <= at < day_start and isinstance(viewers, (int, float)) and viewers == viewers:
            values.append(float(viewers))
    if not values:
        return None
    return max(sum(values) / len(values), 1.0)


def ramp(comp_or_case_rows, day_start, p1):
    """ln(hour 1 / the six pre-midnight hours): how much of the launch
    surge had already happened before the UTC day began."""
    level = pre_level(comp_or_case_rows, day_start)
    if level is None:
        return None
    return math.log(max(p1, 1.0) / level)


def comp_weight(comp, curve, p1, case_ramp):
    """How much one comparable's shape should count for this case."""
    weight = W_UNLOCK[unlock_class(comp)]
    text = ((comp.get("unlock") or "") + " " + (comp.get("name") or "")).lower()
    if "early access" in text or "early-access" in text:
        weight *= W_EARLY_ACCESS_DIRTY if "contaminat" in text else W_EARLY_ACCESS
    if "free-to-play" in text or "free to play" in text or "(f2p" in text:
        weight *= W_F2P
    first = max(curve[1], 1.0)
    peak = max(max(curve.values()), 1.0)
    hot = first / peak
    weight *= HOT_START_FLOOR + (1.0 - HOT_START_FLOOR) * min(1.0, hot / HOT_START_FULL)
    gap = math.log(p1 / first)
    weight *= math.exp(-0.5 * (gap / MAG_SIGMA) ** 2)
    if case_ramp is not None:
        try:
            start = parse_ts(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            start = None
        if start is not None:
            own = ramp(comp.get("hourly"), start, first)
            if own is not None:
                weight *= math.exp(-0.5 * ((case_ramp - own) / RAMP_SIGMA) ** 2)
    return weight


def weighted_quantile(pairs, tau):
    """The tau quantile of (value, weight) pairs, linear on the cumulated
    weight; pairs must be non-empty. Deterministic: sorted by value."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    run = 0.0
    prev_value, prev_run = ordered[0][0], 0.0
    for value, w in ordered:
        run += w
        if run >= target:
            if run == prev_run:
                return value
            frac = (target - prev_run) / (run - prev_run)
            return prev_value + frac * (value - prev_value)
        prev_value, prev_run = value, run
    return ordered[-1][0]


def persistence_row(p1, hour):
    """The house baseline's row for one hour: flat level, widening spread."""
    spread = 0.15 * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z]


def pooled_log_quantiles(case, p1):
    """For each hour, the five pooled log-ratio quantiles (weighted pool
    blended with the unweighted pool), or None where no pool exists."""
    weighted = {h: [] for h in HOURS}
    plain = {h: [] for h in HOURS}
    usable = 0
    case_ramp = None
    try:
        day_start = parse_ts(case["day_start"])
        case_ramp = ramp(case.get("history_hourly") or case.get("history"),
                         day_start, p1)
    except (KeyError, TypeError, ValueError):
        case_ramp = None
    for comp in case.get("comparables") or []:
        if not is_launch(comp):
            continue
        curve = release_day_curve(comp)
        if 1 not in curve or len(curve) < 20:
            continue
        first = max(curve[1], 1.0)
        weight = comp_weight(comp, curve, p1, case_ramp)
        if weight <= 0:
            continue
        usable += 1
        for hour in HOURS:
            if hour in curve:
                ratio = math.log(max(curve[hour], 1.0) / first)
                ratio = max(-RATIO_CAP, min(RATIO_CAP, ratio))
                weighted[hour].append((ratio, weight))
                plain[hour].append((ratio, 1.0))
    if usable < MIN_POOL:
        return None
    rows = {}
    for hour in HOURS:
        if len(weighted[hour]) < MIN_POOL:
            return None
        rows[hour] = [
            POOL_BLEND * weighted_quantile(weighted[hour], tau)
            + (1.0 - POOL_BLEND) * weighted_quantile(plain[hour], tau)
            for tau in LEVELS
        ]
    return rows


def smooth_median(rows):
    """Lightly smooth the pooled log-median across neighbouring hours:
    launch curves bend smoothly, per-hour pools are noisy."""
    med = {h: rows[h][2] for h in HOURS}
    out = {}
    for i, hour in enumerate(HOURS):
        near = [med[HOURS[j]] for j in (i - 1, i, i + 1) if 0 <= j < len(HOURS)]
        out[hour] = sum(near) / len(near)
    for hour in HOURS:
        rows[hour][2] = out[hour]
    return rows


def forecast_case(case):
    """The forecast object for one case: 23 rows of five quantiles."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first.get("viewers") or 0))
    rows = pooled_log_quantiles(case, p1)
    hours = []
    if rows is None:
        for row in case["forecast"]:
            hours.append({"hour": row["hour"], "q": persistence_row(p1, row["hour"])})
    else:
        rows = smooth_median(rows)
        for row in case["forecast"]:
            hour = row["hour"]
            logs = rows[hour]
            median = logs[2]
            if PARAM_BLEND > 0:
                # A lognormal read of the same pool: one sigma per hour,
                # averaged from the interquartile and the 5-95 ranges.
                sigma = ((logs[3] - logs[1]) / (2 * 0.674490)
                         + (logs[4] - logs[0]) / (2 * 1.644854)) / 2
                logs = [
                    (1.0 - PARAM_BLEND) * value
                    + PARAM_BLEND * (median + z * sigma)
                    for value, z in zip(logs, Z)
                ]
            qs = []
            for value in logs:
                log_q = median + SPREAD_CAL * (value - median)
                qs.append(min(1e9, max(0.01, p1 * math.exp(log_q))))
            qs.sort()
            hours.append({"hour": hour, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
