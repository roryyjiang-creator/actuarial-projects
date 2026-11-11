"""Simulate a 30-year monthly short-rate history from a Vasicek model, then recover its parameters.

Vasicek: dr = a (b - r) dt + sigma dW. Sampled every dt years the exact discretization is an AR(1):
    r[k+1] = c + phi * r[k] + e,   phi = exp(-a dt),  c = b (1 - phi),  Var(e) = sigma^2 (1 - phi^2) / (2a)
so ordinary least squares gives a, b and sigma. The history is simulated (true values known), so the fit
can be checked. Run from the project5_interest_rates folder.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
A, B, SIG, R0 = 0.25, 0.03, 0.012, 0.045     # true speed, long-run mean, volatility, starting rate
DT, YEARS = 1 / 12, 30
n = int(YEARS / DT)
pd.set_option("display.width", 200)

phi = np.exp(-A * DT)
sd = SIG * np.sqrt((1 - phi**2) / (2 * A))
r = np.empty(n + 1)
r[0] = R0
for k in range(n):
    r[k + 1] = B + (r[k] - B) * phi + sd * rng.standard_normal()
pd.DataFrame({"month": range(n + 1), "short_rate": r}).to_csv("data/short_rate.csv", index=False)
print(f"simulated {n + 1} monthly short rates: mean {r.mean():.4f}, min {r.min():.4f}, max {r.max():.4f}")

def fit(x):
    """OLS of x[k+1] on x[k]; return Vasicek (a, b, sigma)."""
    X, y = x[:-1], x[1:]
    slope, icpt = np.polyfit(X, y, 1)
    resid = y - (icpt + slope * X)
    a = -np.log(slope) / DT
    b = icpt / (1 - slope)
    s = resid.std(ddof=2) * np.sqrt(2 * a / (1 - slope**2))
    return a, b, s

a_hat, b_hat, s_hat = fit(r)

# bootstrap the rate path by resampling AR(1) residuals, to see how well each parameter is pinned down
X, y = r[:-1], r[1:]
slope, icpt = np.polyfit(X, y, 1)
resid = y - (icpt + slope * X)
boots = []
for _ in range(1000):
    rb = np.empty(n + 1)
    rb[0] = r[0]
    e = rng.choice(resid, n)
    for k in range(n):
        rb[k + 1] = icpt + slope * rb[k] + e[k]
    boots.append(fit(rb))
boots = np.array(boots)
lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

tab = pd.DataFrame({"parameter": ["speed a", "long-run mean b", "volatility sigma"],
                    "true": [A, B, SIG], "estimate": [a_hat, b_hat, s_hat],
                    "95% low": lo, "95% high": hi})
print("\nVASICEK PARAMETERS FROM 30 YEARS OF MONTHLY RATES")
print(tab.round(4).to_string(index=False))
print(f"\nhalf-life of a rate shock: true {np.log(2) / A:.1f} years, estimated {np.log(2) / a_hat:.1f} years")
tab.to_csv("reports/vasicek_params.csv", index=False)
print("saved data/short_rate.csv and reports/vasicek_params.csv")
