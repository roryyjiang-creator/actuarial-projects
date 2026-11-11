import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_poisson_deviance

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

age_labels = {"(17, 25]": "18-25", "(25, 35]": "26-35", "(35, 45]": "36-45",
              "(45, 55]": "46-55", "(55, 65]": "56-65", "(65, 100]": "66+"}
veh_labels = {"(-1, 2]": "0-2", "(2, 5]": "3-5", "(5, 10]": "6-10", "(10, 100]": "11+"}
bm_edges = [0, 50, 60, 75, 90, 100, 110, 125, 150]
bm_labels = ["<=50", "51-60", "61-75", "76-90", "91-100", "101-110", "111-125", "126-150"]

for d in (train, test):
    d["DrivAgeBand"] = d["DrivAgeBand"].map(age_labels)
    d["VehAgeBand"] = d["VehAgeBand"].map(veh_labels)
    d["BMBand"] = pd.cut(d.BonusMalus, bm_edges, labels=bm_labels).astype(str)
    d["VehPowerC"] = d.VehPower.clip(4, 10).astype(int).astype(str)
    assert d["DrivAgeBand"].notna().all() and d["VehAgeBand"].notna().all()
    assert (d["BMBand"] != "nan").all(), "bonus-malus outside the bands"

def T(col):
    base = train.groupby(col).Exposure.sum().idxmax()
    return f"C({col}, Treatment(reference='{base}'))"

formulas = {
    "v1: script 05 model": "ClaimNb ~ " + " + ".join([T("DrivAgeBand"), T("VehAgeBand"), T("Area"), T("VehGas")]) + " + LogDensity + BonusMalus",
    "v1 without BonusMalus": "ClaimNb ~ " + " + ".join([T("DrivAgeBand"), T("VehAgeBand"), T("Area"), T("VehGas")]) + " + LogDensity",
    "v2: banded BM + power + brand + region": "ClaimNb ~ " + " + ".join([T("DrivAgeBand"), T("VehAgeBand"), T("VehGas"), T("BMBand"), T("VehPowerC"), T("VehBrand"), T("Region")]) + " + LogDensity",
}

def gini(y, pred, w):
    o = np.argsort(pred.values)
    cw = np.r_[0, np.cumsum(w.values[o]) / w.sum()]
    cy = np.r_[0, np.cumsum(y.values[o]) / y.sum()]
    return 1 - 2 * np.trapezoid(cy, cw)

null_rate = train.ClaimNb.sum() / train.Exposure.sum()
models, cals, rows = {}, {}, []
for name, f in formulas.items():
    print("fitting:", name, flush=True)
    m = smf.glm(f, data=train, family=sm.families.Poisson(), offset=np.log(train.Exposure)).fit()
    te = test.copy()
    te["rate"] = m.predict(te, offset=np.zeros(len(te)))
    te["pred"] = te["rate"] * te.Exposure
    y = te.ClaimNb / te.Exposure
    dev = mean_poisson_deviance(y, te["rate"], sample_weight=te.Exposure)
    dev0 = mean_poisson_deviance(y, np.full(len(te), null_rate), sample_weight=te.Exposure)
    te["decile"] = pd.qcut(te["rate"].rank(method="first"), 10, labels=False)
    cal = te.groupby("decile").agg(exposure=("Exposure", "sum"), actual=("ClaimNb", "sum"), predicted=("pred", "sum"))
    cal["pred_vs_actual"] = (cal.predicted / cal.actual - 1).round(3)
    models[name], cals[name] = m, cal
    rows.append({"model": name, "coefficients": len(m.params),
                 "deviance_improvement": f"{1 - dev / dev0:.2%}",
                 "gini": round(gini(te.ClaimNb, te["rate"], te.Exposure), 3),
                 "worst_decile_miss": f"{cal.pred_vs_actual.abs().max():.1%}",
                 "pearson_chi2_per_df": round(m.pearson_chi2 / m.df_resid, 2)})

print("\n=== Model comparison (test set) ===")
print(pd.DataFrame(rows).to_string(index=False))

# Does the young-driver result flip when BonusMalus is removed?
age = {}
for name, m in models.items():
    p = m.params[m.params.index.str.contains("DrivAgeBand")]
    p.index = p.index.str.extract(r"\[T\.(.*)\]")[0]
    age[name] = np.exp(p).round(3)
print("\n=== Driver-age relativities (base 46-55) ===")
print(pd.DataFrame(age))

v2 = "v2: banded BM + power + brand + region"
print("\n=== Calibration by decile, v2 ===")
print(cals[v2].round(3))

# Final model: refit with scale='X2' (quasi-Poisson) so confidence intervals allow for overdispersion
final = smf.glm(formulas[v2], data=train, family=sm.families.Poisson(),
                offset=np.log(train.Exposure)).fit(scale="X2")
ci = final.conf_int()
rel = pd.DataFrame({"relativity": np.exp(final.params), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1])}).round(3)
rel.to_csv("reports/frequency_v2_relativities.csv")
print("\nSaved reports/frequency_v2_relativities.csv (quasi-Poisson intervals)")
print(rel.to_string())
