# Project 4: Policyholder Lifecycle Markov Model

A continuous-time Markov chain for a policy's life: **Active**, **Claim** (open claim) and **Lapsed** (absorbing). I simulate 20,000 policies over 10 years, estimate the transition rates, check the model against nonparametric survival curves, fit hazard ratios, and value the premium and claim cash flows with the generator matrix.

## Data
Simulated, not real. Rates differ by a high-risk flag (30% of policies). Because the data come from a Markov chain, agreement between the model and the data is expected here. On real data the same checks would test the memoryless assumption.

## Stages
| Script | What it does |
|---|---|
| 01_simulate_histories.py | Simulates policy histories as spells (state, start, end, next state), censored at 10 years |
| 02_estimate_rates.py | Maximum-likelihood rates = transitions / time in state, with 95% intervals; expected years in each state from -alpha Q^-1; checks against expm(Q t) and the data |
| 03_survival_and_hazard_ratios.py | Kaplan-Meier curves vs the model; hazard ratios from Poisson regression with a time-in-state exposure and a time-varying "open claim" covariate |
| 04_present_value.py | Present value of premiums and claims as a survival-weighted annuity, with sensitivity to interest and lapse rates, and breakeven premium |

## Results
- All 8 estimated rates fall inside their 95% intervals around the true values.
- Expected years in force before lapse: 6.56 (low risk) vs 4.58 (high risk).
- Kaplan-Meier and model retention agree to within about 0.5 percentage points (reports/retention_curves.png).
- Lapse hazard ratio, high vs low risk: 1.41 (1.37 to 1.46; true 1.4). Open claim vs active: 2.57 (2.37 to 2.78; true 2.67). Claim hazard ratio, high vs low risk: 1.95 (1.87 to 2.03; true 2.0).
- Present value (illustrative inputs: premium 1,000/yr, claim cost 5,000, acquisition cost 600, 4% interest): profit per policy about 2,049 (low risk) and 197 (high risk). The matrix formula matches the average over simulated policies to within about 6.
- Sensitivity: 8% interest cuts high-risk profit from 197 to 108. Raising lapse rates 50% turns it negative (-11). Charging high-risk policies the low-risk breakeven premium loses about 1,454 per policy.

## Method notes
- Time in each state is the exposure, so censored policies are handled without special treatment.
- PV over a horizon T: alpha (dI - Q)^-1 (I - exp((Q - dI)T)) 1, where Q is the generator among non-absorbing states and d is the force of interest.
- Hazard ratios use Poisson regression, which equals a piecewise-exponential model because hazards are constant within a state. A Cox model would be the choice if the baseline hazard shape were unknown.

## Limitations
- Simulated data, so model checks are not independent tests.
- Premium, claim cost, acquisition cost and interest are my assumptions, so the dollar figures illustrate the method and are not findings.
- Constant rates within a state and no dependence on duration since policy start. Real lapse rates usually change with policy age.
- No confidence intervals on the PV figures. A policy bootstrap that refits the rates would give them.

## Run
Run each script from this folder, in order:

    python src/01_simulate_histories.py
    python src/02_estimate_rates.py
    python src/03_survival_and_hazard_ratios.py
    python src/04_present_value.py

data/histories.csv is gitignored and is recreated by script 01.
