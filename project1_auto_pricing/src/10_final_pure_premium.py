"""Final pure premium evaluation: v1 vs v2 frequency combined with the severity model.

Fits everything on data/train.csv, scores data/test.csv. Run from project1_auto_pricing.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- helpers ----------
def gini_norm(pred_rate, loss, expo):
    """Exposure-weighted normalized Gini of loss, ordered by predicted rate."""
    def area(order):
        e, l = expo[order], loss[order]
        cx = np.concatenate([[0], np.cumsum(e) / e.sum()])
        cy = np.concatenate([[0], np.cumsum(l) / l.sum()])
        return np.trapezoid(cy, cx)
    g = 1 - 2 * area(np.argsort(pred_rate, kind="stable"))
    g_perfect = 1 - 2 * area(np.argsort(loss / expo, kind="stable"))
    return g / g_perfect

def decile_table(pred_rate, loss, expo):
    order = np.argsort(pred_rate, kind="stable")
    e, l, p = expo[order], loss[order], pred_rate[order] * expo[order]
    dec = np.minimum(np.ceil(np.cumsum(e) / e.sum() * 10), 10).astype(int)
    t = pd.DataFrame({"decile": dec, "exposure": e, "actual": l, "predicted": p}).groupby("decile").sum()
    t["actual_per_yr"] = t.actual / t.exposure
    t["pred_per_yr"] = t.predicted / t.exposure
    t["actual_over_pred"] = t.actual / t.predicted
    return t

# ---------- data ----------
train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
for name, df in (("train", train), ("test", test)):
    for need in ("ClaimNb", "Exposure", "DrivAge", "VehAge", "VehPower", "VehGas", "VehBrand",
                 "Region", "Area", "Density", "BonusMalus", "n_sev", "tot_loss_cap", "tot_loss"):
        if need not in df.columns:
            raise SystemExit(f"data/{name}.csv has no column '{need}'. Columns: {list(df.columns)}")

def prep(df):
    df = df.copy()
    for c in ("VehGas", "VehBrand", "Region", "Area"):
        df[c] = df[c].astype(str).str.strip("'\" ")
    df["DrivAgeBand"] = pd.cut(df.DrivAge, [17, 25, 35, 45, 55, 65, 100],
                               labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]).astype(str)
    df["VehAgeBand"] = pd.cut(df.VehAge, [-1, 2, 5, 10, 100], labels=["0-2", "3-5", "6-10", "11+"]).astype(str)
    df["BMBand"] = pd.cut(df.BonusMalus.clip(upper=150), [0, 50, 60, 75, 90, 100, 110, 125, 150],
                          labels=["<=50", "51-60", "61-75", "76-90", "91-100", "101-110", "111-125", "126-150"]).astype(str)
    df["VehPowerC"] = df.VehPower.clip(4, 10).astype(int).astype(str)
    df["LogDensity"] = np.log(df.Density)
    df["BonusMalusC"] = df.BonusMalus.clip(upper=150)
    return df

train, test = prep(train), prep(test)
expo_te = test.Exposure.to_numpy()
loss_te = test.tot_loss_cap.to_numpy()

# ---------- frequency models ----------
f_v1 = ("ClaimNb ~ C(DrivAgeBand) + C(VehAgeBand) + C(Area) + C(VehGas) + LogDensity + BonusMalusC")
f_v2 = ("ClaimNb ~ C(DrivAgeBand) + C(VehAgeBand) + C(VehGas) + C(BMBand) + C(VehPowerC)"
        " + C(VehBrand) + C(Region) + LogDensity")
rates = {}
for name, f in (("v1", f_v1), ("v2", f_v2)):
    m = smf.glm(f, train, family=sm.families.Poisson(), offset=np.log(train.Exposure)).fit()
    rates[name] = np.asarray(m.predict(test, offset=np.zeros(len(test))))
    print(f"frequency {name}: fitted, predicted claims on test {rates[name].dot(expo_te):,.0f} "
          f"vs actual {test.ClaimNb.sum():,.0f}")

# ---------- severity model (policies with claims, capped losses, claim-count weights) ----------
sev_tr = train[train.n_sev > 0].copy()
sev_tr["avg_sev"] = sev_tr.tot_loss_cap / sev_tr.n_sev
f_sev = "avg_sev ~ C(DrivAgeBand) + C(VehAgeBand) + C(VehGas) + LogDensity"
msev = smf.glm(f_sev, sev_tr, family=sm.families.Gamma(sm.families.links.Log()),
               freq_weights=sev_tr.n_sev).fit()
sev_pred = np.asarray(msev.predict(test))
sev_const = sev_tr.tot_loss_cap.sum() / sev_tr.n_sev.sum()
print(f"severity: Gamma GLM fitted on {len(sev_tr):,} policies; average capped severity {sev_const:,.0f}")

# large-loss load from training data (all losses / capped losses)
load = train.tot_loss.sum() / train.tot_loss_cap.sum()
print(f"large-loss load (train): {load:.3f}")

# ---------- candidate pure premiums (per exposure year) ----------
cands = {
    "constant": np.full(len(test), train.tot_loss_cap.sum() / train.Exposure.sum()),
    "v2 freq x constant severity": rates["v2"] * sev_const,
    "v1 freq x severity model": rates["v1"] * sev_pred,
    "v2 freq x severity model": rates["v2"] * sev_pred,
}

rows = []
for name, pp in cands.items():
    t = decile_table(pp, loss_te, expo_te)
    rows.append({"model": name,
                 "Gini": gini_norm(pp, loss_te, expo_te) if name != "constant" else 0.0,
                 "top/bottom decile actual": t.actual_per_yr.iloc[-1] / t.actual_per_yr.iloc[0],
                 "worst decile miss": (t.actual_over_pred - 1).abs().max(),
                 "total actual/pred": loss_te.sum() / (pp * expo_te).sum()})
res = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 200)
print("\nTest-set comparison (capped losses):")
print(res.round(3).to_string())

best = cands["v2 freq x severity model"]
tab = decile_table(best, loss_te, expo_te)
print("\nLift table, v2 frequency x severity (deciles by predicted pure premium, exposure-weighted):")
print(tab[["exposure", "actual_per_yr", "pred_per_yr", "actual_over_pred"]].round(3).to_string())
print(f"\nTotal pure premium incl. large-loss load: predicted {best.dot(expo_te) * load:,.0f} "
      f"vs actual uncapped {test.tot_loss.sum():,.0f} "
      f"(ratio {test.tot_loss.sum() / (best.dot(expo_te) * load):.3f})")

tab.to_csv("reports/lift_table_v2.csv")
res.to_csv("reports/pure_premium_model_comparison.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(1, 11)
ax.bar(x - 0.2, tab.actual_per_yr, 0.4, label="Actual")
ax.bar(x + 0.2, tab.pred_per_yr, 0.4, label="Predicted")
ax.set_xlabel("Decile of predicted pure premium (1 = lowest risk)")
ax.set_ylabel("Capped loss per exposure year")
ax.set_title("Pure premium lift, v2 frequency x severity (test set)")
ax.set_xticks(x); ax.legend()
fig.tight_layout(); fig.savefig("reports/lift_chart_v2.png", dpi=150)
print("saved reports/lift_table_v2.csv, pure_premium_model_comparison.csv, lift_chart_v2.png")
