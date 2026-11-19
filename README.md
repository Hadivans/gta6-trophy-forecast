# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hadivans/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hadivans/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one. `main.py`
(Python 3.13, standard library only) reads one case `/input/<id>.in` —
every Exophase trophy-page read the house made before the checkpoint —
and writes `/output/<id>.out` with five quantiles (5%, 25%, 50%, 75%,
95%) of each of the game's 55 PlayStation 5 trophies' earn rate at the
case's target (2026-11-26 00:00 UTC for the final). Score is pinball
loss on the logit scale; a prize needs a lower loss than the best house
entry (two baselines and a house AI forecast).

## Method

Per trophy, from the case's own history only:

- Collapse the history to distinct reads: Exophase only changes a
  game's counts at its own refreshes, so consecutive identical reads
  carry no new signal and would overweight stale values.
- Fit logit(rate) against ln(1 + hours since launch) by
  recency-weighted least squares over the last 48 hours of distinct
  reads, with the slope shrunk toward the tier's median slope (story
  bronze trophies move together; grind trophies move together). The
  48-hour window matters: a saturating rate's logit is concave in
  log-time, so the steep launch-night slope would overshoot, while the
  recent slope extrapolates.
- Take the fitted line's own value at the target as the median. The
  spread combines the fit residual, the slope uncertainty over the
  extrapolation distance, a random-walk term that grows with the
  horizon, and a floor for the next refresh (one refresh can move
  every rate at once — on launch night the tracked base jumped from
  638 to 3,868 players and every rate but the one at 100% fell).

Deterministic (two runs write the same bytes), no network, no model
calls, well under the 8-second limit.

## Backtest

The kit's true practice answers are server-side only, so the local
backtest uses synthetic refresh worlds that reproduce the known
structure of the data (refresh-stepped reads, a fast-growing tracked
base, saturating per-trophy curves). Mean pinball logit loss on
final-like cases (checkpoint launch + 59.5h, horizon 4.52 days):

| forecaster | loss | typical miss |
|---|---|---|
| this entry | 0.158 | 1.15 pp |
| house curve baseline | 0.314 | 2.77 pp |
| house persistence baseline | 0.398 | 4.08 pp |

Practice-like cases (90-minute horizon): this entry 1.073 vs 1.142
persistence and 1.301 curve. Honest caveat: the house AI model
cannot be backtested (no data in the kit), so beating the baselines
does not guarantee beating it.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Entry

Built for the [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty `gta6-trophy-rarity-forecast-01` (pot $250, 20 paid places).
