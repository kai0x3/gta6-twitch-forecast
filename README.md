# Hour-by-hour launch-day Twitch forecast — a gta6-twitch-ccu-forecast-01 entry

A make.fun "Forecast GTA VI's launch-day Twitch viewers hour by hour" entry
by @kai0x3. The program reads one case file (everything known up to the end
of hour 1 of a game's release day, including 38–42 past big launches with
their full hourly curves) and writes five quantiles (5/25/50/75/95%) of the
Twitch category's concurrent viewers for each of hours 2–24.

## The method

A launch day's shape, measured from its own first hour, repeats across big
launches. For every comparable with a usable release-day curve the program
takes the log-ratio of each hour's viewers to that launch's hour-1 viewers,
pools the ratios hour by hour with weights for how much each comparable
resembles this case — a rolling midnight-local console unlock like this one,
no early-access head start, a hot start, a similar hour-1 size, and a
similar pre-midnight ramp (hour 1 vs the six hours before the UTC day) —
blends the weighted pool with the plain pool, smooths the median across
neighbouring hours, and reads the five quantiles off a lognormal fit to the
pool. The forecast is `hour-1 print × exp(pooled log-ratio quantile)`. If
too few comparables are usable it falls back to flat persistence with a
widening spread. No network, no randomness: two runs write the same bytes.

Validation: a leave-one-out backtest over the 37 usable comparables (each
in turn becomes a pseudo-case forecast from the others, scored with the
contest's exact log-scale pinball loss) gives **0.585** for this model vs
**0.930** for the house baseline formula on the same footing; over the 12
rolling-midnight launches — the family GTA VI belongs to — **0.245** vs
0.357, with 90% interval coverage of 0.899.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. Each case runs in a fraction of a
second, far under the 8-second limit.
