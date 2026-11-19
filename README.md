# GTA VI trophy rarity forecaster

Forecasts every trophy's Exophase earn rate at the case target, five quantiles
(5/25/50/75/95%) in percent, scored by mean pinball loss on the logit scale.

Model, pure stdlib, no network, deterministic:
- pool dilution: earn rate = earners / players tracked; the tracked pool floods with
  casuals after launch so rates fall. We fit log(players) as a line in log(hours since
  first read) and each trophy's logit(rate) as a line in the same clock, blend the
  trend with strong persistence (weight grows with reads and history span), and damp
  long-horizon extrapolation.
- spread: random-walk term over sqrt(days) plus fit residual plus a pool-growth
  dilution floor, so the 90% band is honestly wide over a 7-day horizon. Coverage is
  scored, hairline bands are how you lose.

Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
Writes <id>.out as {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.
