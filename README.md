# GTA VI trophy rarity forecaster

Reads a case from /input/<id>.in (trophy earn-rate reads over time + target time)
and writes five quantiles per trophy to /output/<id>.out. Pure Python 3 stdlib,
offline and deterministic. Logit-space saturating-exponential curve fit, blended
toward the hold-rate baseline; spread from observed logit volatility.
