"""GTA VI trophy earn-rate forecaster for make.fun gta6-trophy-rarity-forecast-01.
Reads a case file from /input/<id>.in, writes /output/<id>.out.
Forecasts each trophy's earn rate at the target time as five quantiles
(5%, 25%, 50%, 75%, 95%) on the logit scale, minimizing pinball loss.

Model (pure Python stdlib, offline, deterministic):
  * Per trophy: map earn-rate p(t) to logit z = log(p/(1-p)).
  * Smooth the logit path with a saturating exponential approach toward a
    cap: z(t) = Zinf - (Zinf - z0) * exp(-k * (t - t0)), fitted by a coarse
    grid search on k and Zinf from the observed reads.
  * Blends toward the strong house baseline (rates hold) when history is
    short or noisy: z_forecast = w*z_curve + (1-w)*z_last.
  * Quantile spread from observed logit volatility: widen for long horizon,
    for steep/active trophies, and for high overall level.
"""
import json
import math
import os
import sys

INPUT_DIR = "/input"
OUTPUT_DIR = "/output"
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def fit_curve(points):
    """points: list of (t_hours, p). Returns ((z_start,k,Zinf), vol)."""
    if not points:
        return None, None
    ts = [p[0] for p in points]
    zs = [logit(p[1]) for p in points]
    t0 = ts[0]
    rel = [t - t0 for t in ts]

    drifts = []
    for i in range(1, len(points)):
        dt = rel[i] - rel[i - 1]
        if dt > 0:
            drifts.append(abs(zs[i] - zs[i - 1]) / dt)
    vol = median(drifts) if drifts else 0.02

    best = None
    z_start, z_last = zs[0], zs[-1]
    z_hi = max(z_last, max(zs)) + 3.0
    for ki in range(0, 60):
        k = 10 ** (-4.0 + 0.09 * ki)
        for zi in range(0, 40):
            span = z_hi - (z_last - 0.5)
            Z = (z_last - 0.5) + span * zi / 39.0
            if Z < z_last - 1e-9:
                Z = z_last
            sse = 0.0
            for r, z in zip(rel, zs):
                pred = Z - (Z - z_start) * math.exp(-k * r)
                sse += (pred - z) ** 2
            if best is None or sse < best[0]:
                best = (sse, k, Z)
    sse, k, Z = best
    return (z_start, k, Z), vol


def forecast_trophy(points, t_target):
    if not points:
        return 0.0, 0.0, 0.0
    points = sorted(points)
    z_last = logit(points[-1][1])
    t_last = points[-1][0]
    n = len(points)
    horizon = max(t_target - t_last, 0.0)

    fit, vol = fit_curve(points)

    if fit is None or n < 3 or horizon <= 0:
        z_curve = z_last
    else:
        z_start, k, Z = fit
        t0 = points[0][0]
        z_curve = Z - (Z - z_start) * math.exp(-k * (t_target - t0))

    duration = points[-1][0] - points[0][0]
    if n >= 8 and duration >= 12:
        w = 0.65
    elif n >= 5 and duration >= 6:
        w = 0.45
    elif n >= 3:
        w = 0.25
    else:
        w = 0.0
    z_fc = w * z_curve + (1 - w) * z_last
    if n >= 2:
        dt_all = points[-1][0] - points[0][0]
        if dt_all > 0:
            pace = (z_last - logit(points[0][1])) / dt_all
            trend_limit = z_last + pace * horizon * 0.35
        else:
            trend_limit = z_last
    else:
        trend_limit = z_last
    hi = max(z_last, trend_limit) + 0.15
    lo = min(z_last, trend_limit) - 0.10
    z_fc = min(max(z_fc, lo), hi)

    level = sigmoid(z_fc)
    spread = 0.18 + 0.30 * math.sqrt(max(horizon, 0.0)) * (0.4 + 0.6 * min(vol, 0.20) / 0.20)
    spread *= 1.0 + 0.35 * (1.0 - abs(level - 0.5) * 2.0)
    if n < 5:
        spread *= 1.6
    return z_fc, spread, vol


def process_case(case):
    out = {}
    t_target = case.get("target_hours")
    for tr in case.get("trophies", []):
        tid = tr.get("id") or tr.get("name")
        reads = tr.get("reads", []) or tr.get("series", [])
        points = []
        for r in reads:
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                points.append((float(r[0]), float(r[1])))
            elif isinstance(r, dict):
                t = r.get("t", r.get("hours", r.get("time")))
                p = r.get("p", r.get("rate", r.get("earn_rate")))
                if t is not None and p is not None:
                    points.append((float(t), float(p)))
        if not points and "last" in tr:
            points = [(0.0, float(tr["last"]))]
        tgt = t_target if t_target is not None else (points[-1][0] if points else 0)
        z_fc, spread, vol = forecast_trophy(points, tgt)
        qs = [sigmoid(z_fc + spread * (q - 0.5) * 2.2) for q in QUANTILES]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out[str(tid)] = [round(q, 6) for q in qs]
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # prefer /input when it holds cases; fall back to ./input (dev/testing)
    indir = INPUT_DIR
    if not (os.path.isdir(indir) and any(f.endswith(".in") for f in os.listdir(indir))):
        if os.path.isdir("input"):
            indir = "input"
    written = 0
    if os.path.isdir(indir):
        for fname in sorted(os.listdir(indir)):
            if not fname.endswith(".in"):
                continue
            cid = fname[:-3]
            with open(os.path.join(indir, fname), "r") as fh:
                raw = fh.read().strip()
            try:
                case = json.loads(raw)
            except json.JSONDecodeError:
                case = {"trophies": [], "target_hours": None}
            fc = process_case(case)
            with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w") as oh:
                json.dump({"case": cid, "forecast": fc}, oh)
            written += 1
    if written == 0:
        with open(os.path.join(OUTPUT_DIR, "empty.out"), "w") as oh:
            json.dump({"case": "none", "forecast": {}}, oh)


if __name__ == "__main__":
    main()
