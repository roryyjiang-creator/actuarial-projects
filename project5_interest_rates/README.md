# Project 5: Discounting Claim Reserves Under Interest-Rate Scenarios

How much does interest-rate uncertainty matter for the present value of outstanding claim payments? I estimate a claim payment pattern from a loss triangle, fit a Vasicek interest-rate model, discount the outstanding payments, and build the same calculation as an Excel tool.

## Data
Simulated, not real. A 10-year paid-loss triangle with a known payment pattern, and a 30-year monthly short-rate history from a Vasicek model with known parameters. Known truth lets me check each estimate.

## Stages
| Script | What it does |
|---|---|
| 01_triangle_and_pattern.py | Simulates the triangle; chain-ladder age-to-age factors give the payment pattern; projects outstanding payments by future calendar year |
| 02_vasicek_calibrate.py | Simulates the rate history; recovers the Vasicek speed, long-run mean and volatility by AR(1) regression, with bootstrap intervals |
| 03_discount_scenarios.py | Present value under flat rates and Vasicek (closed form checked against 20,000 simulated paths); sensitivity to rates, model parameters and payment timing |
| 04_build_excel.py | Builds excel/reserve_discounting.xlsx with live formulas and a Check sheet against Python |

## Results
- The chain-ladder payment pattern is within 0.3 points of the true one at every development year. The estimated unpaid reserve is 18.87M against a true 20.10M (6.1% low), mostly from the newest accident year.
- 44% of the outstanding reserve is paid next year and 26% the year after; under 1% is paid beyond year 7.
- Vasicek volatility is well estimated (0.0127 vs true 0.012). Speed (0.51 vs 0.25) and long-run mean (3.9% vs 3.0%) are not: their bootstrap intervals miss the true values, a known bias of mean-reversion estimates from short histories.
- Present value: 17.87M under the Vasicek model, 5.3% below the undiscounted 18.87M. The closed-form bond prices match the Monte Carlo average to four decimals.
- Rate paths spread the PV by about 0.9% of the reserve (5th to 95th percentile: 93.1% to 96.3% of undiscounted).
- Starting rate +1 point: -0.9% PV. Payments half a year later: -1.7%. Using the true instead of fitted parameters: +0.7%.
- The Excel workbook reproduces every Python figure; its Check sheet reports all rows OK.

## Method notes
- Chain ladder: age-to-age factor j is total cumulative payments at age j+1 divided by total at age j, over accident years that have both. The payment pattern is the reciprocal of the product of the remaining factors. Tail factor assumed 1.0.
- Vasicek: dr = a(b - r)dt + sigma dW. Zero-coupon price P(0,t) = A(t) exp(-B(t) r), with B(t) = (1 - exp(-a t)) / a. Sampled monthly the model is an AR(1), so OLS estimates a, b and sigma.
- Payments are assumed to fall mid-year.

## Limitations
- Simulated data, so the calibration check is not available on real rates. Real series (for example FRED Treasury rates) could replace the simulated history.
- Chain ladder with no tail beyond year 9; discounting uses expected payments only, so claim-amount uncertainty is not modelled.
- Vasicek allows negative rates and has one factor. Hull-White or CIR would address that.
- The reserve is short-tailed, so rate risk looks small; a longer-tailed line would be more sensitive.
- The Excel Check sheet compares against fixed Python values and does not update when inputs change. Formulas were verified by recalculating in LibreOffice and in Excel.

## Run
Run each script from this folder, in order:

    python src/01_triangle_and_pattern.py
    python src/02_vasicek_calibrate.py
    python src/03_discount_scenarios.py
    python src/04_build_excel.py
