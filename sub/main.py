"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads a case file, writes /output/<id>.out as {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}
with quantiles in PERCENT on the levels 0.05, 0.25, 0.50, 0.75, 0.95, scored by mean pinball
loss on the logit scale.

What moves an Exophase earn rate over a week:
  rate(t) = earners(t) / players_tracked(t)
both rise, but the tracked pool floods with casuals after launch, so rates fall.
We model the two counts separately and divide:

  * players(t): the tracked pool. Fit log(players) linear in log(hours since the first
    read); the pool grows like a power law after a launch, so this is near exact.
    Clamp the daily growth to a sane band to avoid blow-ups on 2 reads.
  * earners(t) = rate(t) * players(t) / 100, read off every historical read. Earners
    only grow. Fit earners as a *fraction of the pool*: f(t) = rate. The fraction
    falls as casuals arrive; fit logit(rate) linear in log(hours since launch), so a
    trophy with a real decay trend extrapolates downward, a flat one stays put.
  * blend the trend with strong persistence: median_logit = w*trend + (1-w)*last_logit,
    w rising with the number of reads and the span of the history.
  * spread (logit units): honest width. A 7-day horizon over a launch week is a regime
    change, not noise, so the spread grows with sqrt(days) and is floor-ed high when
    the pool is still growing fast. Coverage is scored, so it must be real, not hairline.

Pure stdlib, offline, deterministic, fast (thousands of reads x 55 trophies in << 8s).
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def linfit(xs, ys):
    """Least squares slope/intercept; returns (slope, intercept, resid_std)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter = my - slope * mx
    resid = [y - (slope * x + inter) for x, y in zip(xs, ys)]
    std = math.sqrt(sum(r * r for r in resid) / n)
    return slope, inter, std


def parse_time(s):
    # "2026-11-19T03:00:00Z" or with millis; use a tiny manual parse (stdlib datetime ok too)
    try:
        from datetime import datetime, timezone
        t = s.replace("Z", "+00:00")
        return datetime.fromisoformat(t).timestamp()
    except Exception:
        return 0.0


def forecast_case(case):
    trophies = case["trophies"]
    history = [h for h in case.get("history", []) if h.get("rates")]
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    target_ts = parse_time(case.get("target", ""))
    out_rows = []
    if not history:
        return {"trophies": []}

    # launch epoch: first read we know of; hours since it are the trend clock
    t0 = parse_time(history[0].get("at", "")) or 0.0
    # target hours since t0 (fallback: last read + horizon)
    last_ts = parse_time(history[-1].get("at", ""))
    if target_ts:
        target_h = (target_ts - t0) / 3600.0
    else:
        target_h = (last_ts - t0) / 3600.0 + horizon_days * 24.0

    # --- players pool trend: log(players) ~ a + b*log1p(hours) ---
    ph = []
    pl = []
    for h in history:
        p = h.get("players_tracked")
        ts = parse_time(h.get("at", ""))
        if p and ts:
            ph.append(math.log1p(max((ts - t0) / 3600.0, 0.0)))
            pl.append(math.log(p))
    pool_slope, pool_inter, pool_resid = (0.0, math.log(638.0), 0.0)
    if len(ph) >= 2:
        pool_slope, pool_inter, pool_resid = linfit(ph, pl)
        pool_slope = min(max(pool_slope, -0.5), 1.5)  # sane daytime growth band
    # pool growth rate per day near the last read (for the spread floor)
    last_h = max((last_ts - t0) / 3600.0, 0.0) if last_ts else 0.0
    denom = max(1.0 + last_h, 1e-6)
    pool_growth_per_day = abs(pool_slope) / denom * 24.0

    nreads = len(history)
    span_days = max((last_ts - t0) / 86400.0, 0.0) if last_ts and t0 else 0.0
    if nreads >= 8 and span_days >= 1.0:
        w = 0.75
    elif nreads >= 5 and span_days >= 0.4:
        w = 0.55
    elif nreads >= 3:
        w = 0.30
    else:
        w = 0.0

    for idx, tr in enumerate(trophies):
        series = []  # (hours_since_t0, rate_pct)
        for h in history:
            rates = h.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                ts = parse_time(h.get("at", ""))
                if ts:
                    series.append(((ts - t0) / 3600.0, float(rates[idx])))
        if not series:
            # no data at all: flat at a tiny rate with a wide honest spread
            zc, spread = logit(0.5), 1.2
        else:
            last_rate = series[-1][1]
            z_last = logit(last_rate)
            if len(series) >= 3:
                xs = [math.log1p(max(h, 0.0)) for h, _ in series]
                ys = [logit(r) for _, r in series]
                slope, inter, resid = linfit(xs, ys)
                # long-horizon extrapolation damping: flatten the slope far out
                slope = max(slope, -1.6)  # rates can fall, but not off a cliff
                xt = math.log1p(max(target_h, 0.0))
                z_trend = inter + slope * xt
                # the trend must not resurrect a dead trophy or flatten a fast one
                z_trend = min(max(z_trend, z_last - 3.0), z_last + 1.0)
                z_med = w * z_trend + (1.0 - w) * z_last
                if abs(z_med - z_last) > 2.4:
                    z_med = z_last + math.copysign(2.4, z_med - z_last)
            else:
                z_med, resid = z_last, 0.3
            zc = z_med
            # spread: base random walk + fit residual + pool-dilution risk
            spread = 0.35 + 0.55 * math.sqrt(horizon_days)
            spread += min(resid * 1.2, 0.8)
            spread += min(pool_growth_per_day * horizon_days * 0.6, 0.9)
            if len(series) < 5:
                spread *= 1.35
            spread = min(spread, 2.6)
        qs = [round4(100.0 * sigmoid(zc + zz * spread)) for zz in Z]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        qs = [min(max(q, 0.0), 100.0) for q in qs]
        out_rows.append({"id": tr["id"], "q": qs})
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    if INPUT_DIR.is_dir():
        for f in sorted(INPUT_DIR.iterdir()):
            if not f.name.endswith(".in"):
                continue
            cid = f.name[:-3]
            case = json.loads(f.read_text())
            result = forecast_case(case)
            with open(OUTPUT_DIR / (cid + ".out"), "w") as oh:
                json.dump(result, oh, separators=(",", ":"))
            written += 1
    if written == 0:
        with open(OUTPUT_DIR / "empty.out", "w") as oh:
            json.dump({"trophies": []}, oh)


if __name__ == "__main__":
    main()
