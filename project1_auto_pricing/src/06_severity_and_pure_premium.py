import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_gamma_deviance

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

age_labels = {"(17, 25]": "18-25", "(25, 35]": "26-35", "(35, 45]": "36-45",
              "(45, 55]": "46-55", "(55, 65]": "56-65", "(65, 100]": "66+"}
veh_labels = {"(-1, 2]": "0-2", "(2, 5]": "3-5", "(5, 10]": "6-10", "(10, 100]": "11+"}
for d in (train, test):
    d["DrivAgeBand"] = d["DrivAgeBand"].map(age_labels)
    d["VehAgeBand"] = d["VehAgeBand"].map(veh_labels)

def base(col):
    return train.groupby(col).Exposure.sum().idxmax()

def terms_for(factors):
    return [f"C({c}, Treatment(reference='{base(c)}'))" for c in factors]

# ---------- Frequency model (same as script 05) ----------
freq_formula = "ClaimNb ~ " + " + ".join(terms_for(["DrivAgeBand", "VehAgeBand", "Area", "VehGas"])) + " + LogDensity + BonusMalus"
freq = smf.glm(freq_formula, data=train, family=sm.families.Poisson(),
               offset=np.log(train.Exposure)).fit()

# ---------- Severity model ----------
# Policies with at least one claim; target = average capped claim size,
# weighted by the number of claims so policies with more claims count more.
sev_train = train[train.n_sev > 0].copy()
sev_test = test[test.n_sev > 0].copy()
print("policies with claims: train", len(sev_train), " test", len(sev_test))

sev_formula = "avg_sev ~ " + " + ".join(terms_for(["DrivAgeBand", "VehAgeBand", "VehGas"])) + " + LogDensity"
sev = smf.glm(sev_formula, data=sev_train,
              family=sm.families.Gamma(sm.families.links.Log()),
              freq_weights=sev_train.n_sev).fit()
print(sev.summary())

ci = sev.conf_int()
rel = pd.DataFrame({"relativity": np.exp(sev.params), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1])}).round(3)
rel.to_csv("reports/severity_relativities.csv")
print("\nSeverity relativities:\n", rel)

# Severity accuracy on test policies with claims, versus a constant average
sev_test["sev_pred"] = sev.predict(sev_test)
const = sev_train.tot_loss_cap.sum() / sev_train.n_sev.sum()
d_model = mean_gamma_deviance(sev_test.avg_sev, sev_test.sev_pred, sample_weight=sev_test.n_sev)
d_const = mean_gamma_deviance(sev_test.avg_sev, np.full(len(sev_test), const), sample_weight=sev_test.n_sev)
print(f"\nTest Gamma deviance: model={d_model:.4f}  constant={d_const:.4f}  improvement={(1 - d_model / d_const):.2%}")

# ---------- Pure premium = frequency x severity ----------
test["freq_rate"] = freq.predict(test, offset=np.zeros(len(test)))   # claims per exposure-year
test["sev_pred"] = sev.predict(test)
test["pure_prem"] = test.freq_rate * test.sev_pred                    # capped loss per exposure-year

# Large-loss load: restores the dollars removed by the 100,000 cap
load = train.tot_loss.sum() / train.tot_loss_cap.sum()
test["pure_prem_loaded"] = test.pure_prem * load
print(f"\nLarge-loss load factor (train): {load:.3f}")

def gini(y, pred, w):
    o = np.argsort(pred.values)
    cw = np.r_[0, np.cumsum(w.values[o]) / w.sum()]
    cy = np.r_[0, np.cumsum(y.values[o]) / y.sum()]
    return 1 - 2 * np.trapezoid(cy, cw)

print(f"Test Gini (capped losses, ordered by predicted pure premium): {gini(test.tot_loss_cap, test.pure_prem, test.Exposure):.3f}")

# Lift table: predicted vs actual pure premium by decile
test["decile"] = pd.qcut(test.pure_prem.rank(method="first"), 10, labels=False)
lift = test.groupby("decile").apply(lambda d: pd.Series({
    "exposure": d.Exposure.sum(),
    "predicted": np.average(d.pure_prem, weights=d.Exposure),
    "actual": d.tot_loss_cap.sum() / d.Exposure.sum()}), include_groups=False)
print("\nLift by decile (capped loss per exposure-year):\n", lift.round(1))
lift.to_csv("reports/lift_table.csv")

print("\nTotals on test set: predicted", round((test.pure_prem * test.Exposure).sum()),
      " actual", round(test.tot_loss_cap.sum()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ax = lift[["predicted", "actual"]].plot(marker="o")
ax.set_xlabel("Predicted-risk decile (0 = lowest)")
ax.set_ylabel("Capped loss per exposure-year")
ax.set_title("Lift chart: predicted vs actual pure premium")
plt.savefig("reports/lift_chart.png", dpi=150)
print("saved reports/lift_chart.png")
