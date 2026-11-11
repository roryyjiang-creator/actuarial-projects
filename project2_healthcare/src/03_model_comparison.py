"""Compare cost models with repeated 10-fold CV, scored in dollars.

Models: OLS, log-OLS (with smearing correction), Ridge, Lasso, Gamma GLM (log link).
Each is run on main effects and with smoker x BMI interaction terms.
Run from the project2_healthcare folder.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/insurance.csv").drop_duplicates().reset_index(drop=True)
print(f"{len(df)} rows after dropping exact duplicates")
y = df.charges.to_numpy()
df["smoker_bmi"] = (df.smoker == "yes") * df.bmi
df["smoker_obese"] = ((df.smoker == "yes") & (df.bmi >= 30)).astype(float)

CAT = ["sex", "smoker", "region"]
NUM = ["age", "bmi", "children"]
FEATURES = {"main effects": NUM, "+ smoker x BMI": NUM + ["smoker_bmi", "smoker_obese"]}

def prep(num_cols):
    return ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT),
                              ("num", StandardScaler(), num_cols)])

class SmearedLogOLS(BaseEstimator, RegressorMixin):
    """OLS on log(charges); predictions exp(fit) x mean(exp(residual)) (Duan smearing)."""
    def fit(self, X, y):
        self.lm_ = LinearRegression().fit(X, np.log(y))
        self.smear_ = np.mean(np.exp(np.log(y) - self.lm_.predict(X)))
        return self
    def predict(self, X):
        return np.exp(self.lm_.predict(X)) * self.smear_

class NaiveLogOLS(SmearedLogOLS):
    """Same fit but exp(fit) with no smearing, to show the bias."""
    def fit(self, X, y):
        super().fit(X, y); self.smear_ = 1.0
        return self

MODELS = {
    "OLS": LinearRegression(),
    "log-OLS, no correction": NaiveLogOLS(),
    "log-OLS, smearing": SmearedLogOLS(),
    "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 30)),
    "Lasso": LassoCV(cv=5, max_iter=50_000),
    "Gamma GLM (log link)": GammaRegressor(alpha=0, max_iter=1000),
}

def cv_scores(model, X, y, reps=3, k=10):
    rmse, mae, oof0 = [], [], None
    for rep in range(reps):
        oof = np.empty(len(y))
        for tr, te in KFold(k, shuffle=True, random_state=rep).split(X):
            m = clone(model).fit(X.iloc[tr], y[tr])
            oof[te] = m.predict(X.iloc[te])
            rmse.append(np.sqrt(np.mean((y[te] - oof[te]) ** 2)))
            mae.append(np.mean(np.abs(y[te] - oof[te])))
        if rep == 0:
            oof0 = oof
    return np.array(rmse), np.array(mae), oof0

smoker = (df.smoker == "yes").to_numpy()
rows = []
for fname, cols in FEATURES.items():
    X = df[CAT + NUM + [c for c in cols if c not in NUM]]
    for mname, est in MODELS.items():
        pipe = make_pipeline(prep(cols), est)
        rmse, mae, oof = cv_scores(pipe, X, y)
        rows.append({"features": fname, "model": mname,
                     "RMSE": rmse.mean(), "RMSE sd": rmse.std(), "MAE": mae.mean(),
                     "pred/actual smokers": oof[smoker].mean() / y[smoker].mean(),
                     "pred/actual non-smokers": oof[~smoker].mean() / y[~smoker].mean(),
                     "min pred": oof.min()})
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\n10-fold CV x 3 repeats, dollar scale (RMSE/MAE are means over 30 folds):")
print(res.round(3).to_string(index=False))
res.to_csv("reports/model_comparison.csv", index=False)
print("\nsaved reports/model_comparison.csv")
