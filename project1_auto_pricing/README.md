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