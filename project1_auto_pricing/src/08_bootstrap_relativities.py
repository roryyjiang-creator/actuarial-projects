import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

SEED, N_SAMPLE, N_BOOT = 42, 100_000, 200
rng = np.random.default_rng(SEED)

train = pd.read_csv("data/train.csv")
sample = train.sample(n=min(N_SAMPLE, len(train)), random_state=SEED).reset_index(drop=True)

sample["DrivAgeBand"] = pd.cut(sample["DrivAge"], [17, 25, 35, 45, 55, 65, 100],
                               labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"])
sample["BMBand"] = pd.cut(sample["BonusMalus"].clip(upper=150),
                          [0, 50, 60, 75, 90, 100, 110, 125, 150],
                          labels=["<=50", "51-60", "61-75", "76-90", "91-100",
                                  "101-110", "111-125", "126-150"])
sample["LogDensity"] = np.log(sample["Density"])
sample["VehGas"] = sample["VehGas"].astype(str).str.strip("'\" ")
sample["DrivAgeBand"] = sample["DrivAgeBand"].astype(str)
sample["BMBand"] = sample["BMBand"].astype(str)

formula = ("ClaimNb ~ C(DrivAgeBand, Treatment(reference='46-55'))"
           " + C(VehGas, Treatment(reference='Regular'))"
           " + C(BMBand, Treatment(reference='<=50')) + LogDensity")
y, X = dmatrices(formula, sample, return_type="dataframe")
offset = np.log(sample["Exposure"].to_numpy())
yv, Xv = y.iloc[:, 0].to_numpy(), X.to_numpy()
names = list(X.columns)

targets = {n: i for i, n in enumerate(names)
           if "DrivAgeBand" in n and "18-25" in n or "VehGas" in n and "Diesel" in n}

def fit(idx):
    return sm.GLM(yv[idx], Xv[idx], family=sm.families.Poisson(),
                  offset=offset[idx]).fit()

full_idx = np.arange(len(sample))
base = fit(full_idx)
qp = sm.GLM(yv, Xv, family=sm.families.Poisson(), offset=offset).fit(scale="X2")
print(f"Sample: {len(sample):,} policies, {int(yv.sum()):,} claims; "
      f"Pearson chi2/df = {base.pearson_chi2 / base.df_resid:.2f}")

boot = np.empty((N_BOOT, len(names)))
for b in range(N_BOOT):
    idx = rng.integers(0, len(sample), len(sample))
    boot[b] = fit(idx).params
    if (b + 1) % 25 == 0:
        print(f"  bootstrap {b + 1}/{N_BOOT}")

rows = []
ci_qp = qp.conf_int()
for n, i in targets.items():
    lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
    rows.append({"term": n, "relativity": np.exp(base.params[i]),
                 "boot_lo": np.exp(lo), "boot_hi": np.exp(hi),
                 "quasi_lo": np.exp(ci_qp[i, 0]), "quasi_hi": np.exp(ci_qp[i, 1])})
out = pd.DataFrame(rows)
out["boot_width"] = out.boot_hi - out.boot_lo
out["quasi_width"] = out.quasi_hi - out.quasi_lo
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
out.to_csv("reports/bootstrap_relativities.csv", index=False)
print("saved reports/bootstrap_relativities.csv")

# Robust (sandwich) check on the same sample
rob = sm.GLM(yv, Xv, family=sm.families.Poisson(), offset=offset).fit(cov_type="HC0")
ci_rob = rob.conf_int()
print("\nRobust (HC0) 95% intervals:")
for n, i in targets.items():
    print(f"{n[:45]:45s} {np.exp(ci_rob[i, 0]):.3f} to {np.exp(ci_rob[i, 1]):.3f}")
