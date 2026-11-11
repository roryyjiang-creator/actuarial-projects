# Project 2: Healthcare cost model with validation

Predict individual medical charges from age, sex, BMI, children, smoker status and region,
with validation first, cross-validated model comparison, and bootstrap intervals.
Data: a public copy of the widely used `insurance.csv` teaching dataset (1,338 rows);
it is not real insurer claims data.

## Run order (from this folder)
    python src/01_download_data.py
    python src/02_validate.py
    python src/03_model_comparison.py
    python src/04_gamma_check_and_bootstrap.py

## Data validation
No missing values, no out-of-range age, BMI, children or charges, consistent category labels.
One exact duplicate pair (rows 195 and 581, identical down to charges of 1,639.5631); I dropped
one copy because matching to four decimals is implausible for two different people, leaving
1,337 rows. The decision is a judgment call with negligible effect on results.

## The cost distribution
Mean 13,270 vs median 9,382 (ratio 1.41); max 63,770, only about 4.8x the mean. The top 10%
of people account for 32% of charges. Smokers are 20.5% of people but 49.5% of charges
(mean 32,050 vs 8,434). Most of the skew is a mixture of two groups, not a heavy tail.

## Model comparison (10-fold CV x 3 repeats, dollar scale)
| Features | Model | RMSE | MAE |
|---|---|---|---|
| main effects | OLS | 6,077 | 4,216 |
| main effects | Gamma GLM (log link) | 7,793 | 4,445 |
| + smoker x BMI | OLS | 4,408 | 2,366 |
| + smoker x BMI | Ridge / Lasso | 4,407 / 4,409 | 2,369 / 2,368 |
| + smoker x BMI | Gamma GLM | 7,841 | 4,163 |
| + smoker x BMI + smoker x age | Gamma GLM | 4,848 | 2,852 |

- The smoker x BMI interaction matters far more than the model family: OLS RMSE falls 27%
  (6,077 to 4,408) and MAE 44%.
- Ridge and Lasso match OLS: with about 1,300 rows and ten predictors there is nothing to shrink.
- The Gamma GLM lost until I added smoker x age (RMSE 7,841 to 4,848). OLS did not change
  with that term, so the age slope in dollars is similar across groups (roughly additive),
  which a log link only reproduces with an explicit interaction.
- Log-OLS needs a smearing correction to undo retransformation bias. One global factor
  helped non-smokers but over-predicted smokers by 21%, because it assumes equal residual
  spread across groups.
- Main-effects OLS predicts negative costs (minimum -2,290); log-link models cannot.

## Bootstrap (1,000 resamples)
| Group | Statistic | Estimate | 95% interval |
|---|---|---|---|
| All | median | 9,386 | 8,799 - 9,870 |
| All | 90th percentile | 34,833 | 31,404 - 36,951 |
| Non-smokers | median | 7,346 | 7,047 - 7,742 |
| Smokers | median | 34,456 | 29,523 - 36,194 |
| Smokers | 90th percentile | 46,643 | 44,641 - 47,446 |

The smokers' median is less stable than their 90th percentile (standard error 4.1% vs 1.4%
of the estimate). This is likely because smokers' costs form two clusters (BMI 30+ vs below)
and the median sits between them.

## Limitations
Small teaching dataset; one random CV scheme; no real claims, exposure or time dimension;
interactions were chosen after looking at the data, so the CV gain is somewhat optimistic.
