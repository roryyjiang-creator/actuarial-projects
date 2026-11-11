import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_poisson_deviance

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Give the age bands readable labels
age_labels = {"(17, 25]": "18-25", "(25, 35]": "26-35", "(35, 45]": "36-45",
              "(45, 55]": "46-55", "(55, 65]": "56-65", "(65, 100]": "66+"}
veh_labels = {"(-1, 2]": "0-2", "(2, 5]": "3-5", "(5, 10]": "6-10", "(10, 100]": "11+"}
for d in (train, test):
    d["DrivAgeBand"] = d["DrivAgeBand"].map(age_labels)
    d["VehAgeBand"] = d["VehAgeBand"].map(veh_labels)
    assert d["DrivAgeBand"].notna().all() and d["VehAgeBand"].notna().all(), "band label mismatch"

# Base level for each factor = the level with the most exposure
def base(col):
    return train.groupby(col).Exposure.sum().idxmax()

factors = ["DrivAgeBand", "VehAgeBand", "Area", "VehGas"]
bases = {c: base(c) for c in factors}
print("Base levels:", bases)

terms = [f"C({c}, Treatment(reference='{bases[c]}'))" for c in factors]
formula = "ClaimNb ~ " + " + ".join(terms) + " + LogDensity + BonusMalus"

glm = smf.glm(formula, data=train, family=sm.families.Poisson(),
              offset=np.log(train.Exposure)).fit()
print(glm.summary())

# Relativities: exp(coefficient) with 95% confidence intervals
ci = glm.conf_int()
rel = pd.DataFrame({"relativity": np.exp(glm.params),
                    "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1])}).round(3)
rel.to_csv("reports/frequency_relativities.csv")
print("\nRelativities:\n", rel)

# Evaluate on the test set: annual claim rate, then expected claims = rate x exposure
test["rate"] = glm.predict(test, offset=np.zeros(len(test)))
test["pred_claims"] = test["rate"] * test.Exposure

null_rate = train.ClaimNb.sum() / train.Exposure.sum()
dev_model = mean_poisson_deviance(test.ClaimNb / test.Exposure, test["rate"], sample_weight=test.Exposure)
dev_null = mean_poisson_deviance(test.ClaimNb / test.Exposure, np.full(len(test), null_rate), sample_weight=test.Exposure)
print(f"\nTest Poisson deviance: model={dev_model:.4f}  constant-rate={dev_null:.4f}  "
      f"improvement={(1 - dev_model / dev_null):.2%}")

def gini(y, pred, w):
    o = np.argsort(pred.values)
    cw = np.r_[0, np.cumsum(w.values[o]) / w.sum()]
    cy = np.r_[0, np.cumsum(y.values[o]) / y.sum()]
    return 1 - 2 * np.trapezoid(cy, cw)

print(f"Test Gini (claims, ordered by predicted rate): {gini(test.ClaimNb, test['rate'], test.Exposure):.3f}")

# Calibration by predicted-risk decile: actual vs predicted claims
test["decile"] = pd.qcut(test["rate"].rank(method="first"), 10, labels=False)
cal = test.groupby("decile").agg(exposure=("Exposure", "sum"), actual=("ClaimNb", "sum"), predicted=("pred_claims", "sum"))
cal["actual_rate"] = cal.actual / cal.exposure
cal["pred_rate"] = cal.predicted / cal.exposure
print("\nCalibration by decile:\n", cal.round(4))
