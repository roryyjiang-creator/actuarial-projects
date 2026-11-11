"""(A) Test why the log-link models lose: add smoker x age. (B) Bootstrap median and 90th percentile.

Run from the project2_healthcare folder.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/insurance.csv").drop_duplicates().reset_index(drop=True)
y = df.charges.to_numpy()
smk = df.smoker == "yes"
df["smoker_bmi"] = smk * df.bmi
df["smoker_obese"] = (smk & (df.bmi >= 30)).astype(float)
df["smoker_age"] = smk * df.age

CAT = ["sex", "smoker", "region"]
BASE = ["age", "bmi", "children", "smoker_bmi", "smoker_obese"]
SETS = {"smoker x BMI": BASE, "smoker x BMI + smoker x age": BASE + ["smoker_age"]}

def pipe(num_cols, est):
    prep = ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT),
                              ("num", StandardScaler(), num_cols)])
    return make_pipeline(prep, est)

def cv(model, X, y, reps=3, k=10):
    rmse, mae, oof0 = [], [], None
    for rep in range(reps):
        oof = np.empty(len(y))
        for tr, te in KFold(k, shuffle=True, random_state=rep).split(X):
            oof[te] = clone(model).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            rmse.append(np.sqrt(np.mean((y[te] - oof[te]) ** 2)))
            mae.append(np.mean(np.abs(y[te] - oof[te])))
        if rep == 0:
            oof0 = oof
    return np.mean(rmse), np.mean(mae), oof0

print("PART A: does adding smoker x age fix the log-link model?")
rows = []
for sname, cols in SETS.items():
    X = df[CAT + cols]
    for mname, est in (("OLS", LinearRegression()), ("Gamma GLM", GammaRegressor(alpha=0, max_iter=1000))):
        rmse, mae, oof = cv(pipe(cols, est), X, y)
        rows.append({"features": sname, "model": mname, "RMSE": rmse, "MAE": mae,
                     "pred/actual smokers": oof[smk].mean() / y[smk].mean(),
                     "pred/actual non-smokers": oof[~smk].mean() / y[~smk].mean()})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nPART B: bootstrap of the median and 90th percentile (1,000 resamples)")
rng = np.random.default_rng(42)
groups = {"all": y, "non-smokers": y[~smk.to_numpy()], "smokers": y[smk.to_numpy()]}
out = []
for name, v in groups.items():
    idx = rng.integers(0, len(v), size=(1000, len(v)))
    samples = v[idx]
    for stat, fn in (("median", lambda a: np.median(a, axis=1)),
                     ("90th pct", lambda a: np.quantile(a, 0.9, axis=1))):
        b = fn(samples)
        point = np.median(v) if stat == "median" else np.quantile(v, 0.9)
        out.append({"group": name, "n": len(v), "statistic": stat, "estimate": point,
                    "boot SE": b.std(ddof=1), "CI low": np.percentile(b, 2.5),
                    "CI high": np.percentile(b, 97.5)})
res = pd.DataFrame(out)
print(res.round(0).astype({"n": int}).to_string(index=False))
res.to_csv("reports/bootstrap_quantiles.csv", index=False)
print("\nsaved reports/bootstrap_quantiles.csv")
