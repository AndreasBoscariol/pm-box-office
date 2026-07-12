# Box Office Production Model

This document describes the production opening-weekend forecast layer used by
the web app and refresh worker. The model emits append-only forecast records for
the Friday-Sunday domestic opening weekend and for the daily Friday, Saturday,
and Sunday components.

## Forecast Timeline

The model supports two forecast phases:

- Pre-release origins: `P_-14` through `P_-1`.
- Live weekend origins: `FRI_10:00` through `SUN_EOD` at two-hour checkpoints
  plus end-of-day checkpoints.

Live forecasts are generated only for origins available by `as_of_utc`. The
active artifact version is selected by `models/boxoffice/ACTIVE_MODEL`.

## Pre-Release Model

For movie `i` and pre-release origin `o`, the point forecast is selected from
the pre-release panel by the frozen point policy:

```text
OW_hat_i,o = selected_point_policy(row_i,o)
```

The current policy uses the configured raw consensus reference column for the
origin. Intervals are selected in this order:

1. Conditional empirical log-residual quantiles from the frozen interval policy.
2. Origin-level empirical log-residual quantiles.
3. Selected interval columns already present in the panel.
4. A fallback log-normal interval using the origin-level sigma policy.

Daily Friday, Saturday, and Sunday component points are allocated from the
opening-weekend point using the row's daily shares when available, otherwise the
default shares:

```text
Friday = 0.42 * OW_hat
Saturday = 0.34 * OW_hat
Sunday = 0.24 * OW_hat
```

When opening-weekend intervals are available, daily component intervals are
allocated by the same component share.

## Live Weekend Baseline

Live weekend forecasts compose opening weekend from daily components. Each daily
component is one of:

- An actual daily gross already known under the live regime.
- An adjusted daily baseline component.
- An optional AMC same-day nowcast for the current target day.

The daily baseline columns are:

| Regime | Known actuals | Baseline components |
| --- | --- | --- |
| `live_friday` | none | `pre_fri_usd`, `pre_sat_usd`, `pre_sun_usd` |
| `live_saturday` | Friday | `after_fri_sat_usd`, `after_fri_sun_usd` |
| `live_sunday` | Friday, Saturday | `after_sat_sun_usd` |

If a day should be known by regime but the actual is missing, production uses the
pre-weekend component as a pending-actual fallback and labels the component with
`actual_missing`.

The deterministic component center is:

```text
OW_center_i,r,tau = Fri_component_i,r,tau
                  + Sat_component_i,r,tau
                  + Sun_component_i,r,tau
```

The reported live opening-weekend point forecast is the median of the simulated
weekend distribution, not necessarily the arithmetic sum of component medians.

## Opening-Thursday Actual Update

Before live components are built, production may update the opening-weekend
prior using an official The Numbers opening-Thursday daily gross. This is the
first production prior update attempted.

The update is eligible only when:

- The frozen `opening_thursday_actual_ratio_update_prod` policy is present and
  enabled.
- The daily gross is positive and sourced from The Numbers.
- The daily gross date is the Thursday immediately before the Friday opening
  weekend start.
- The source record is available by execution time.
- There are no conflicting opening-Thursday actuals.

If eligible, the prior is updated by the frozen ratio model:

```text
x = log(OpeningThursdayGross / OW_baseline)
OW_updated = OW_baseline * exp(alpha + beta * x)
```

The pending live daily components are then rescaled to match `OW_updated`.
Friday/Saturday/Sunday pre-weekend components are scaled together. After-Friday
Saturday/Sunday components are scaled to preserve the known Friday actual when
available. After-Saturday Sunday is scaled to preserve known Friday and Saturday
actuals when available.

If the opening-Thursday actual update is missing, disabled, ineligible, or
invalid, production keeps the baseline prior and records the fallback reason.
AMC-imputed Thursday values are not allowed through this production path.

## Reported Thursday Preview Update

If the opening-Thursday actual update does not apply, production may update the
opening-weekend prior from a reported Thursday preview gross.

The preview update is eligible only when:

- The frozen preview policy is present.
- The baseline opening-weekend prior is positive.
- A positive reported preview gross exists.
- The baseline timestamp is before the preview cutoff timestamp.
- The frozen `alpha` and `beta` coefficients are valid.

The update uses:

```text
OW_updated = OW_baseline * exp(alpha + beta * log(PreviewGross / OW_baseline))
```

The pending live daily components are rescaled in the same way as the
opening-Thursday actual update. If the update is not eligible, the model retains
the baseline consensus prior and records the fallback reason.

## AMC Same-Day Plug-In

The AMC model is an optional same-day plug-in layer. It does not replace the
weekend model. It can replace only the current live target day:

| Regime | AMC target day |
| --- | --- |
| `live_friday` | Friday |
| `live_saturday` | Saturday |
| `live_sunday` | Sunday |

The plug-in row must match the movie, regime, forecast origin, and target day,
and must provide at least:

```text
movie_id
regime
forecast_origin
target_day
pred_daily_gross_usd
sigma_log_daily
source
```

Additional fields such as `model` and `feature_quality_bucket` are preserved in
component metadata when present.

The live component selection rule is:

```text
G_hat_prod(i,d,r,tau) =
  actual gross, if the day is known under the regime or marked available as of tau
  AMC same-day nowcast, if d is the regime target day and a valid plug-in row exists
  adjusted daily baseline, otherwise
```

Therefore:

- `live_friday`: AMC Friday + pre-weekend Saturday + pre-weekend Sunday, or the
  fallback described below when no AMC/prior update is available.
- `live_saturday`: actual Friday + AMC Saturday + after-Friday Sunday.
- `live_sunday`: actual Friday + actual Saturday + AMC Sunday.

If no valid AMC row exists, the model falls back to the adjusted daily baseline
component.

## Friday No-AMC Carry-Forward

For `live_friday`, when there is no AMC component and neither the
opening-Thursday actual update nor the reported preview update applied, the
opening-weekend forecast carries forward the latest eligible pre-release
opening-weekend forecast from `P_-1` or `P_-2`.

This avoids replacing a calibrated pre-release opening-weekend forecast with a
component-sum live baseline when Friday live information has not actually
arrived. If no eligible carry-forward row exists, production raises an error
instead of emitting an unsupported Friday no-AMC live forecast.

## Component Intervals

Each non-actual component receives uncertainty from the best available frozen
policy:

- AMC components use AMC empirical interval cells when available.
- Opening-Friday AMC components are floored by the generic daily interval policy
  so provisional Friday AMC uncertainty cannot become narrower than the generic
  policy.
- Baseline components use daily empirical interval cells by regime and day.
- If empirical policy payloads are unavailable, the model falls back to daily
  log-normal sigma values from the artifact manifest.

Actual components have zero component uncertainty.

## Matched AMC Forward Validation

Whenever an eligible AMC component is used, the refresh worker also writes an
immutable `amc_no_amc_control` emission for the same origin, actual vintage,
baseline artifacts, and fixed market grid. The control differs only by retaining
the state-appropriate non-AMC daily baseline. It never replaces the primary
forecast row.

After resolved weekends, run:

```sh
.venv/bin/python -m models.boxoffice.forward_amc_distribution --model-version latest
```

Immediately after the first eligible AMC refresh, run the read-only canary:

```sh
.venv/bin/python -m models.boxoffice.forward_amc_distribution --model-version latest --canary
```

This writes the append-only paired panel, score summaries, clustered bootstrap,
tail cases, and integrity audit under `data/diagnostics/forward_amc_distribution`.
AMC is operationally enabled but remains under forward statistical validation.

## Weekend Simulation

Opening-weekend intervals are simulated from the daily components with `n_sim`
draws from the artifact manifest, defaulting to 50,000.

For each simulation draw `b` and component day `d`:

```text
G_draw(b,d) =
  G_actual(d), for actual components
  G_hat(d) * exp(residual_draw), for forecast components
```

When empirical residual samples are available, residuals are drawn from those
samples, using frozen sample weights when present. Otherwise residuals are drawn
from a normal distribution using the component sigma.

For live Sunday baseline Sunday components, production may use a signed
Saturday-conditioned Sunday residual policy. It conditions on:

```text
log(actual Saturday / after-Friday Saturday forecast)
```

and blends local residual weights with pooled weights by the frozen shrinkage
parameter.

The simulated opening weekend is:

```text
OW_draw(b) = Fri_draw(b) + Sat_draw(b) + Sun_draw(b)
```

The emitted opening-weekend forecast is:

```text
point_usd = Q_0.50(OW_draw)
lo80_usd = Q_0.10(OW_draw)
hi80_usd = Q_0.90(OW_draw)
lo95_usd = Q_0.025(OW_draw)
hi95_usd = Q_0.975(OW_draw)
```

The interval model label records which path was used:

- `amc_empirical_component_residual_simulation`
- `signed_saturday_conditioned_sunday_residual_simulation`
- `empirical_component_residual_simulation`
- `component_log_error_simulation`

## Persistence and Idempotency

Forecast refreshes write append-only records to:

- `analytics.movie_forecast_emissions`
- `analytics.movie_forecast_emission_components`

Each emission is keyed by model version, release run, origin, target, `as_of_utc`,
and forecast role. The payload hash makes emission writes idempotent. If the same
logical key is written with a different payload hash, production records the
conflict and rejects the write instead of silently mutating the prior emission.

Actuals are attached later as outcomes only. Outcome attachment does not rewrite
the emitted point or interval fields.

## Production Policy Summary

| Regime | Prior update attempts | Known actuals | AMC target | Opening-weekend construction |
| --- | --- | --- | --- | --- |
| `live_friday` | Opening-Thursday actual, then reported preview | none | Friday | AMC Friday + adjusted Sat/Sun, or latest pre-release carry-forward when no AMC/prior update exists |
| `live_saturday` | Opening-Thursday actual, then reported preview | Friday | Saturday | actual Friday + AMC/baseline Saturday + adjusted after-Friday Sunday |
| `live_sunday` | Opening-Thursday actual, then reported preview | Friday, Saturday | Sunday | actual Friday + actual Saturday + AMC/baseline Sunday |

The production model is intentionally conservative: it uses official actuals
when they are known, applies leakage-safe prior updates only when eligibility is
satisfied, limits AMC to the current live target day, and falls back to frozen
baseline or carry-forward behavior whenever live signals are unavailable.
