"""Simulate a paid-loss triangle, estimate the claim payment pattern with chain ladder,
and project the outstanding payments by future calendar year.

Rows are accident years (oldest first), columns are development years 0..9 (years since the accident year).
Only the upper-left triangle is observed. Run from the project5_interest_rates folder.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
N = 10
TRUE_CUM = np.array([0.35, 0.62, 0.78, 0.88, 0.94, 0.97, 0.985, 0.993, 0.997, 1.0])   # share paid by end of dev year j
TRUE_INC = np.diff(np.r_[0, TRUE_CUM])
pd.set_option("display.width", 200)

# ultimate loss per accident year: 10M growing 4% a year, with noise
ult = 10_000_000 * 1.04 ** np.arange(N) * rng.lognormal(0, 0.05, N)
inc = ult[:, None] * TRUE_INC[None, :] * rng.lognormal(0, 0.06, (N, N))      # noisy incremental payments
cum = inc.cumsum(axis=1)
obs = np.full((N, N), np.nan)
for i in range(N):
    obs[i, : N - i] = cum[i, : N - i]            # AY i has N - i development years observed

# ---- chain ladder: volume-weighted age-to-age factors ----
f = np.array([np.nansum(obs[: N - j - 1, j + 1]) / np.nansum(obs[: N - j - 1, j]) for j in range(N - 1)])
cdf = np.r_[np.cumprod(f[::-1])[::-1], 1.0]       # development factor from age j to ultimate (tail = 1.0)
est_cum = 1 / cdf                                  # estimated cumulative payment pattern
est_inc = np.diff(np.r_[0, est_cum])

pat = pd.DataFrame({"dev year": range(N), "true cumulative": TRUE_CUM, "estimated cumulative": est_cum,
                    "estimated incremental": est_inc})
print("1. PAYMENT PATTERN: share of ultimate paid by the end of each development year")
print(pat.round(3).to_string(index=False))

# ---- project the lower triangle ----
proj = obs.copy()
for i in range(N):
    for j in range(N - i, N):
        proj[i, j] = proj[i, j - 1] * f[j - 1]
latest = np.array([obs[i, N - 1 - i] for i in range(N)])
ultimate_est = proj[:, -1]
unpaid_est = ultimate_est - latest
unpaid_true = cum[:, -1] - latest      # payments that actually occur in the simulated future
res = pd.DataFrame({"accident year": range(1, N + 1), "paid to date": latest, "est. ultimate": ultimate_est,
                    "est. unpaid": unpaid_est, "true unpaid": unpaid_true})
print("\n2. RESERVE BY ACCIDENT YEAR")
print(res.round(0).astype({"accident year": int}).to_string(index=False))
print(f"\nTotal unpaid: estimated {unpaid_est.sum():,.0f}, true {unpaid_true.sum():,.0f} "
      f"({unpaid_est.sum() / unpaid_true.sum() - 1:+.1%})")

# ---- outstanding payments by future calendar year (1 = the year after the latest diagonal) ----
cf = np.zeros(N - 1)
for i in range(N):
    for j in range(N - i, N):
        cf[j - (N - i) ] += proj[i, j] - proj[i, j - 1]
cfdf = pd.DataFrame({"year from now": range(1, N), "expected payment": cf})
cfdf["share of reserve"] = cfdf["expected payment"] / cf.sum()
print("\n3. OUTSTANDING PAYMENTS BY FUTURE CALENDAR YEAR (the cash flow we will discount)")
print(cfdf.round(3).to_string(index=False))

pat.to_csv("reports/payment_pattern.csv", index=False)
cfdf.to_csv("reports/outstanding_cashflows.csv", index=False)
print("\nsaved reports/payment_pattern.csv and reports/outstanding_cashflows.csv")
