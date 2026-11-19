"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Method (per trophy, computed only from the case's own history):
- collapse the history to distinct reads: Exophase only changes a
  game's counts at its own refreshes, so consecutive identical reads
  carry no new signal and would overweight stale values;
- fit logit(rate) against ln(1 + hours since launch) by recency-weighted
  least squares over the last 48 hours of distinct reads, with the
  slope shrunk toward the tier's median slope (story bronze trophies
  move together; grind trophies move together). The 48-hour window
  matters: a saturating rate's logit is concave in log-time, so the
  steep launch-night slope would overshoot, while the recent slope
  extrapolates; a quadratic fit was tried and rejected (its
  extrapolation is unstable past the data);
- take the fitted line's own value at the target as the median; the
  spread combines the fit residual, the slope uncertainty over the
  extrapolation distance, a random-walk term that grows with the
  horizon, and a floor for the next refresh (one refresh can move
  every rate at once, as launch night showed: 638 players tracked
  became 3,868 and every rate but the one at 100% fell).

Calibrated on synthetic refresh worlds (see backtest.py in the work
dir): final-like mean pinball logit loss 0.158 vs 0.314 for the house
curve baseline and 0.400 for persistence; practice-like 1.098 vs
1.51 for the house baselines on the real practice (their published
figure). The kit's true practice answers are server-side only, so
the real practice grade is only known after submission; the house AI
model cannot be backtested at all, so beating the baselines does not
guarantee beating it.

Deterministic, standard library only, no network, no model calls.
Reads every <id>.in in the input directory, writes <id>.out.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Origin of the log-time axis: the game's launch day (matches the house
# curve baseline's origin, 2026-11-19T00:00:00Z).
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Tunables, calibrated on synthetic refresh worlds (see backtest.py in the
# work dir; the kit's true practice answers are server-side only).
RECENCY_HALFLIFE_H = 48.0  # recency weight halves every 48 hours
SHRINK_K = 0.5  # ridge weight (in x-variance units) pulling a slope to its tier prior
QUAD_RIDGE = 0.5  # ridge penalty on the quadratic term, as a share of total weight
SPREAD_FLOOR = 0.4  # logit units: no forecast is more certain than this
SPREAD_HORIZON = 0.20  # random-walk spread per sqrt(day) of horizon
SPREAD_CAP = 1.8  # widest spread allowed, in logit units
RESID_FLOOR = 0.15  # smallest residual spread credited to a fit
MAX_LOGIT_MOVE = 2.5  # cap on how far the median may move from the last read
# Shape of the trend fit (swept by backtest.py; the values below are the
# chosen ones). FIT_WINDOW_H: only reads this many hours before the last
# read take part in the fit (None = all of them). FIT_QUAD: fit a
# quadratic in log-time (growth slowing down) instead of a line.
# ANCHOR: anchor the forecast at the last read (True) or take the
# fitted curve's own value at the target (False). DAMP: multiply the
# fitted move by this (growth slows, so a fitted move overstates).
FIT_WINDOW_H = 48.0
FIT_QUAD = False
ANCHOR = False
DAMP = 1.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(t: datetime) -> float:
    return max(0.0, (t - LAUNCH).total_seconds() / 3600.0)


def distinct_reads(history: list) -> list:
    """Collapse consecutive identical reads to their first occurrence.

    Exophase refreshes a game's counts hours apart, so between refreshes
    every read is the same page again. Keeping only the first read of
    each distinct page stops the flat stretches from dominating the fit.
    """
    out = []
    prev = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev:
            out.append(read)
            prev = key
    return out


def wls(xs: list, ys: list, ws: list) -> tuple:
    """Recency-weighted least squares line; returns (intercept, slope)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        return my, 0.0
    slope = sxy / sxx
    return my - slope * mx, slope


def wls_quad(xs: list, ys: list, ws: list, ridge_c: float) -> tuple:
    """Weighted least squares for y ~ a + b*x + c*x^2, with a ridge
    penalty (ridge_c * sum(w)) on c only, so the fit shrinks toward a
    straight line when the data does not support curvature. Returns
    (a, b, c). Solved by Gaussian elimination on the 3x3 system."""
    sw = sum(ws)
    # Standardize x for conditioning: u = (x - mx) / sx.
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    vx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs)) / sw
    sx = math.sqrt(vx) if vx > 1e-12 else 1.0
    us = [(x - mx) / sx for x in xs]
    # Normal equations for [a, b, c] in u.
    m = [[0.0] * 3 for _ in range(3)]
    v = [0.0] * 3
    basis = [[1.0, u, u * u] for u in us]
    for w, phi, y in zip(ws, basis, ys):
        for i in range(3):
            v[i] += w * phi[i] * y
            for j in range(3):
                m[i][j] += w * phi[i] * phi[j]
    m[2][2] += ridge_c * sw
    # Gaussian elimination with partial pivoting.
    n = 3
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            for k in range(col, n + 1):
                a[r][k] -= f * a[col][k]
    sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = a[i][n] - sum(a[i][j] * sol[j] for j in range(i + 1, n))
        sol[i] = s / a[i][i]
    a0, b0, c0 = sol
    # Back to raw x: y = a0 + b0*(x-mx)/sx + c0*((x-mx)/sx)^2.
    c = c0 / (sx * sx)
    b = b0 / sx - 2 * c0 * mx / (sx * sx)
    a = a0 - b0 * mx / sx + c0 * mx * mx / (sx * sx)
    return a, b, c


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = distinct_reads(case["history"])
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_launch(target))
    horizon_days = case["horizon_minutes"] / 1440.0
    last = history[-1] if history else None
    last_time = parse_time(last["at"]) if last else target

    # Per-trophy series in (x, y) with recency weights, and tier slopes.
    # FIT_WINDOW_H keeps only the recent reads in the fit: a saturating
    # rate's logit is concave in log-time, so the recent slope is the
    # one that extrapolates, not the steep launch-night slope.
    series = []
    tier_slopes: dict = {}
    for i, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            t = parse_time(read["at"])
            age_h = max(0.0, (last_time - t).total_seconds() / 3600.0)
            if FIT_WINDOW_H is not None and age_h > FIT_WINDOW_H:
                continue
            xs.append(math.log1p(hours_since_launch(t)))
            ys.append(logit(rate))
            ws.append(max(0.1, 0.5 ** (age_h / RECENCY_HALFLIFE_H)))
        series.append((xs, ys, ws))
        if len(xs) >= 3:
            _, slope = wls(xs, ys, ws)
            tier_slopes.setdefault(trophy["tier"], []).append(slope)
    # Tier prior slopes: the median slope of the tier, shrunk toward 0
    # (a tier with few trophies, like the single platinum, borrows the
    # global median instead of trusting one noisy line).
    all_slopes = sorted(s for ss in tier_slopes.values() for s in ss)
    global_prior = all_slopes[len(all_slopes) // 2] if all_slopes else 0.0
    tier_prior = {}
    for tier, slopes in tier_slopes.items():
        slopes = sorted(slopes)
        med = slopes[len(slopes) // 2]
        w = len(slopes) / (len(slopes) + 4.0)
        tier_prior[tier] = w * med + (1 - w) * global_prior

    rows = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws = series[i]
        last_rate = last["rates"][i] if last and last["rates"][i] is not None else None
        if not xs or last_rate is None:
            # No usable history for this trophy: stay near the middle of
            # the pack with a wide spread.
            med_logit, spread = logit(1.0), 1.2
        else:
            y_last = logit(last_rate)
            x_last = xs[-1]
            sw = sum(ws)
            mx = sum(w * x for w, x in zip(ws, xs)) / sw
            my = sum(w * y for w, y in zip(ws, ys)) / sw
            sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
            prior = tier_prior.get(trophy["tier"], global_prior)
            if len(xs) >= 3:
                intercept, slope = wls(xs, ys, ws)
                # Shrink the slope toward the tier prior: the less
                # x-variance the trophy's own reads carry, the more it
                # borrows from its tier.
                shrink = sxx / (sxx + SHRINK_K)
                slope = shrink * slope + (1 - shrink) * prior
                resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
                s_resid = math.sqrt(sum(r * r for r in resid) / max(1, len(resid) - 2))
                se_slope = s_resid / math.sqrt(sxx) if sxx > 0 else 0.0
                # The fitted value at the target, from the shrunk line.
                line_value = my + slope * (x_target - mx)
                if FIT_QUAD and len(xs) >= 6:
                    quad = wls_quad(xs, ys, ws, QUAD_RIDGE)
                    if quad is not None:
                        a, b, c = quad
                        quad_value = a + b * x_target + c * x_target * x_target
                        quad_last = a + b * x_last + c * x_last * x_last
                        if ANCHOR:
                            move = (quad_value - quad_last) * DAMP
                            med_logit = y_last + move
                        else:
                            med_logit = quad_value
                        resid = [y - (a + b * x + c * x * x) for x, y in zip(xs, ys)]
                        s_resid = math.sqrt(sum(r * r for r in resid) / max(1, len(resid) - 3))
                    else:
                        med_logit = (y_last + (line_value - (my + slope * (x_last - mx))) * DAMP) if ANCHOR else line_value
                else:
                    if ANCHOR:
                        med_logit = y_last + (line_value - (my + slope * (x_last - mx))) * DAMP
                    else:
                        med_logit = line_value
            else:
                move = prior * (x_target - x_last) * DAMP
                med_logit = y_last + move
                s_resid, se_slope = RESID_FLOOR, 0.0
            move = med_logit - y_last
            move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, move))
            med_logit = y_last + move
            dx = max(0.0, x_target - x_last)
            spread = math.sqrt(
                max(s_resid, RESID_FLOOR) ** 2
                + (se_slope * dx) ** 2
                + (SPREAD_HORIZON * math.sqrt(horizon_days)) ** 2
                + SPREAD_FLOOR ** 2
            )
            spread = min(SPREAD_CAP, spread)
        row = [round4(min(99.99, max(0.01, inv_logit(med_logit + z * spread)))) for z in Z]
        # Keep the quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append({"id": trophy["id"], "q": row})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
