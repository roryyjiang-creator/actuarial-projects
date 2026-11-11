# Project 1: Auto Insurance Frequency-Severity Pricing

Goal: build a pure premium model (frequency x severity) on the French motor
third-party liability data (freMTPL2freq and freMTPL2sev from OpenML).

## Reproducing the data

Run `python src/01_download_data.py`, which downloads the data into `data/`.
The data files are not committed to the repository because of their size.

## Data validation

The policy table has 678,013 rows and the claims table has 26,639 rows.
I checked the data before modeling and made these decisions:

| Check | Result | Decision |
|---|---|---|
| Duplicate policy IDs | 0 | None needed |
| Exposure outside (0, 1] | 1,224 policies | Clip exposure at 1 |
| Claim count above 4 | 9 policies | Cap claim count at 4 |
| Claims with no matching policy | 195 claims | Drop them; they have no exposure or rating variables |
| Claim count above 0 but no claim record | 9,116 policies | Define claim count from the claims table so frequency and severity are consistent |
| Claim record but claim count 0 | 0 | None needed |
| Non-positive claim amounts | 0 | None needed |
| VehGas values contained stray quote characters | All rows | Stripped the quotes when loading |

Claims were aggregated to one row per policy before joining to the policy table,
to avoid double-counting exposure. Loss dollars were reconciled between the
claims table and the model table.

## Data preparation

Each claim is capped at 100,000 (currency units, likely euros) before aggregating to the
policy level. The cap removes 17.7% of total loss dollars (59.9M to 49.3M), so
a large-loss load of about 1.215 (total losses / capped losses) is needed to recover the
full cost. Handling the large-loss load is outside this model and noted as a limitation.

The data is split 75/25 into train and test sets (random_state=42). Frequency is
similar in both (0.0735 vs 0.0744 claims per exposure-year). Average severity differs
by about 5% (1,887 vs 1,797), which I attribute to the small number of claims and
skewed amounts in the test set.

## Results
## Frequency model (Poisson GLM, exposure offset)

Poisson GLM with log(exposure) as offset; predictors: driver age band, vehicle age
band, area, fuel type, log density, bonus-malus. Test Poisson deviance improves 3.86%
over a constant rate; test Gini 0.274. Strongest predictor: bonus-malus (+2.7% per
point). Diesel +20%. Area is not significant once density is included.

Findings: (1) driver age relativities below 1 for young drivers, likely because
bonus-malus already captures claims history; to be checked by refitting without it.
(2) Pearson chi-square / df is about 1.6, suggesting overdispersion, so the confidence
intervals are probably too narrow. (3) Calibration by decile is good in total (1.2%)
but under-predicts deciles 6-7 by 17-25% and over-predicts deciles 0-1 by about 13%,
so the model is missing structure (candidates: non-linear bonus-malus, vehicle power,
region, brand).

## Severity model (Gamma GLM)
Severity (Gamma GLM, log link, claim-count weights): rating variables add almost nothing.
Test Gamma deviance improves 0.35% over a constant average. Only the 18-25 driver
band stands out (relativity 1.18, 95% CI 1.01 to 1.38); with ten coefficients
tested I treat it as weak evidence.

Pure premium (frequency x severity, capped at 100,000): test Gini 0.295. Predicted
and actual pure premium rise together across risk deciles, with noise in two
middle deciles. Total predicted capped loss on the test set is 3.6% above actual
(12.41M vs 11.98M), consistent with the lower average severity seen in the test
split. A large-loss load of 1.228 (train total / capped losses) restores the
dollars removed by the cap.

Limitations: no trend or credibility adjustment; claim sizes are very dispersed so
the Gamma assumption is rough; the large-loss load is a single flat factor.

## Frequency model v2

Adding banded bonus-malus, vehicle power, brand and region (and dropping area, which
overlaps with density) improves the test Poisson deviance from 3.86% to 4.70% and the
Gini from 0.274 to 0.294. The worst decile miss falls from 19.8% to 9.3% (predicted
vs actual); total predicted claims are 1.3% below actual.

Check on the young-driver result: with bonus-malus in the model the 18-25 relativity is
0.785, but removing bonus-malus changes it to 2.03. Bonus-malus carries most of the
claims-history risk, so age effects here are conditional on it.

Overdispersion (Pearson chi-square / df about 1.67) persists in every model, so the
final confidence intervals use a quasi-Poisson fit. Region relativities are noisy
where few policies exist (for example R43: 0.67 to 1.67), which would motivate
regularization or credibility weighting.

## Uncertainty check: bootstrap vs quasi-Poisson

On a 100,000-policy sample (3,863 claims, Pearson chi2/df = 1.53), I compared 200-resample
percentile bootstrap intervals with quasi-Poisson intervals for two relativities:

| Term | Relativity | Bootstrap 95% | Quasi-Poisson 95% |
|---|---|---|---|
| Age 18-25 vs 46-55 | 0.824 | 0.737-0.916 | 0.691-0.983 |
| Diesel vs Regular | 1.156 | 1.090-1.229 | 1.068-1.252 |

Conclusions agree (Diesel is higher-frequency; young drivers are below base once
bonus-malus is in the model). The bootstrap intervals are narrower than quasi-Poisson,
so quasi-Poisson is the conservative choice for reported intervals. Next check: robust
(sandwich) standard errors and more resamples.

Robust (sandwich, HC0) intervals fall between the other two methods, and match the
Poisson standard errors almost exactly for these terms, so the overdispersion (1.53)
does not come from the age or fuel effects. I report the robust intervals:

| Term | Relativity | Robust 95% |
|---|---|---|
| Age 18-25 vs 46-55 | 0.824 | 0.717-0.948 |
| Diesel vs Regular | 1.156 | 1.082-1.235 |

The quasi-Poisson intervals (0.691-0.983 and 1.068-1.252) are conservative for these
terms because one scale factor is applied to every coefficient. The bootstrap (200
resamples) is slightly narrower than the robust interval; more resamples would stabilise it.

## Excel rating calculator

`excel/auto_pricing_calculator.xlsx` (built by `src/09_build_excel_calculator.py`) turns the
fitted GLMs into a rating tool:

- **Rel** sheet: one row per model term (key = model|variable|level) with relativity and CI.
- **Calculator** sheet: inputs for driver age, vehicle age/power, fuel, brand, region, density,
  and bonus-malus. `XLOOKUP` pulls each relativity; frequency x severity x large-loss load
  (1.228) gives the expected loss cost per policy.
- **Reconciliation** sheet: 20,000 held-out test policies scored with the same Excel
  formulas, summarised by bonus-malus band (actual vs predicted claims).

Reconciliation result: 788 actual claims vs 772.7 predicted (ratio 1.020), matching an
independent Python calculation. A 2% gap is within sampling noise (about 3.6% on 788 claims),
so no rebalancing of the base rate was applied.

Notes: relativities are saved at 6 decimals because continuous terms (e.g. log density) are
raised to large powers and 3-decimal rounding distorts the results. Severity uses only
driver age, vehicle age, fuel and density, so those are the only severity factors in the tool.

By bonus-malus band, actual/predicted ranged 0.97 to 1.12 in the five bands with 68 or more claims; the three highest bands (31 claims combined) are too thin to judge.

## Final pure premium evaluation

Frequency (Poisson GLM, exposure offset) x severity (Gamma GLM, capped at 100,000, weighted
by claim count), fitted on the training split and scored on the held-out test split
(`src/10_final_pure_premium.py`).

| Model | Gini | Top/bottom decile loss | Worst decile miss |
|---|---|---|---|
| v1 frequency x severity | 0.300 | 5.1x | 21.5% |
| v2 frequency x constant severity | 0.312 | 5.5x | 23.4% |
| v2 frequency x severity model | 0.314 | 5.7x | 17.1% |

- v2 improves risk ranking over v1 (Gini 0.314 vs 0.300). The severity model adds little
  (0.002 Gini), consistent with its 0.35% deviance gain: frequency drives the pricing.
- Actual loss per exposure year rises from 65 in the lowest predicted decile to 371 in the
  highest (5.7x); the top decile generates about 28% of capped loss.
- Decile-level calibration is noisy for loss data: an unranked constant model shows 20.8%
  decile misses, so v2's 17.1% is within sampling noise.
- Total capped loss is 3.3% below prediction for every model including the constant, so the
  gap comes from lower loss in the test sample, not from the model. Including the large-loss
  load (1.228 from training) the ratio is 0.926, because the test sample's uncapped-to-capped
  ratio was about 1.18. The load is driven by a few very large claims and would need
  multi-year estimation and smoothing in practice.

Limitations: one random split, not out-of-time; no expenses, trend or profit loading;
losses capped at 100,000 with a flat load rather than a fitted large-loss model.


*Validate with a similar model on google
