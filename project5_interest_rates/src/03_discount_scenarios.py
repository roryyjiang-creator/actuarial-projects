"""Present value of the outstanding claim payments under interest-rate scenarios.

(1) flat rates, (2) Vasicek closed-form bond prices checked against Monte Carlo, (3) the spread of PV across
simulated rate paths, (4) sensitivity to the fitted parameters and to the payment timing.
Payments fall mid-year. Run from the project5_interest_rates folder.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
pd.set_option("display.width", 200)

cf = pd.read_csv("reports/outstanding_cashflows.csv")
pay = cf["expected payment"].to_numpy()
t = cf["year from now"].to_numpy() - 0.5            # mid-year payment times
tot = pay.sum()
prm = pd.read_csv("reports/vasicek_params.csv").set_index("parameter")
a_hat, b_hat, s_hat = prm.loc["speed a", "estimate"], prm.loc["long-run mean b", "estimate"], prm.loc["volatility sigma", "estimate"]
a_tru, b_tru, s_tru = prm.loc["speed a", "true"], prm.loc["long-run mean b", "true"], prm.loc["volatility sigma", "true"]
r0 = pd.read_csv("data/short_rate.csv").short_rate.iloc[-1]       # latest observed short rate

def bond(tt, r, a, b, s):
    """Vasicek zero-coupon bond price P(0, tt) given short rate r."""
    B = (1 - np.exp(-a * tt)) / a
    lnA = (B - tt) * (a * a * b - s * s / 2) / (a * a) - s * s * B * B / (4 * a)
    return np.exp(lnA - B * r)

print(f"outstanding payments, undiscounted: {tot:,.0f}; latest short rate r0 = {r0:.4f}\n")

# ---- 1. flat rates ----
rows = []
for y in (0.0, 0.02, 0.04, 0.06):
    df = np.exp(-y * t)
    pv = (pay * df).sum()
    mac = (t * pay * df).sum() / pv
    rows.append({"flat rate": f"{y:.0%}", "PV": pv, "discount": 1 - pv / tot, "avg time to payment (yrs)": mac})
print("1. FLAT-RATE PRESENT VALUE")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- 2. Vasicek: closed form vs Monte Carlo ----
P = bond(t, r0, a_hat, b_hat, s_hat)
pv_cf = (pay * P).sum()
M, DT = 20000, 1 / 12
steps = int(round(t.max() / DT)) + 1
phi = np.exp(-a_hat * DT)
sd = s_hat * np.sqrt((1 - phi**2) / (2 * a_hat))
r = np.full(M, r0)
integral = np.zeros(M)
disc_at = {}
for k in range(steps):
    r_new = b_hat + (r - b_hat) * phi + sd * rng.standard_normal(M)
    integral += 0.5 * (r + r_new) * DT
    r = r_new
    disc_at[k + 1] = np.exp(-integral)
idx = [int(round(x / DT)) for x in t]
DF = np.column_stack([disc_at[i] for i in idx])        # discount factors per path and payment
pv_paths = DF @ pay
print("\n2. VASICEK PRESENT VALUE: closed form vs Monte Carlo (20,000 rate paths)")
print(pd.DataFrame({"payment year": cf["year from now"], "closed-form P(0,t)": P, "Monte Carlo mean": DF.mean(axis=0)}).round(4).to_string(index=False))
print(f"PV closed form {pv_cf:,.0f}; Monte Carlo mean {pv_paths.mean():,.0f} "
      f"(standard error {pv_paths.std() / np.sqrt(M):,.0f})")

# ---- 3. spread across paths ----
flat_r0 = (pay * np.exp(-r0 * t)).sum()
flat_b = (pay * np.exp(-b_hat * t)).sum()
q = np.percentile(pv_paths, [5, 50, 95])
vals = np.array([pv_paths.mean(), q[0], q[1], q[2], pv_paths.std(), flat_r0, flat_b])
names = ["mean", "5th percentile", "median", "95th percentile", "std dev", "flat at r0", "flat at long-run mean b"]
print("\n3. PV ACROSS RATE PATHS")
print(pd.DataFrame({"measure": names, "PV": vals, "as % of undiscounted": vals / tot}).round(3).to_string(index=False))

# ---- 4. sensitivity ----
rows = []
scen = {"fitted parameters": (a_hat, b_hat, s_hat, r0),
        "true parameters": (a_tru, b_tru, s_tru, r0),
        "fitted, r0 + 1%": (a_hat, b_hat, s_hat, r0 + 0.01),
        "fitted, r0 - 1%": (a_hat, b_hat, s_hat, r0 - 0.01),
        "fitted, long-run mean +1%": (a_hat, b_hat + 0.01, s_hat, r0)}
for label, (a, b, s, r_start) in scen.items():
    rows.append({"scenario": label, "PV": (pay * bond(t, r_start, a, b, s)).sum()})
print("\n4a. SENSITIVITY TO THE RATE MODEL (closed form)")
d = pd.DataFrame(rows); d["vs fitted"] = (d.PV / pv_cf - 1).round(4); d["PV"] = d.PV.round(0)
print(d.to_string(index=False))

rows = []
for shift in (-0.5, 0.0, 0.5, 1.0):
    ts = t + shift
    rows.append({"payments shifted by (yrs)": shift, "PV": (pay * bond(ts, r0, a_hat, b_hat, s_hat)).sum()})
d = pd.DataFrame(rows); d["vs base"] = (d.PV / pv_cf - 1).round(4); d["PV"] = d.PV.round(0)
print("\n4b. SENSITIVITY TO PAYMENT TIMING (all payments earlier or later)")
print(d.to_string(index=False))

# ---- chart ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].hist(pv_paths / tot, bins=60, color="tab:blue", alpha=0.8)
ax[0].axvline(pv_paths.mean() / tot, color="black", lw=1.5, label="mean over paths")
ax[0].axvline(flat_r0 / tot, color="tab:red", ls="--", label="flat at current rate")
ax[0].set_xlabel("PV as share of undiscounted reserve"); ax[0].set_ylabel("rate paths"); ax[0].legend()
ax[0].set_title("PV of outstanding payments, 20,000 Vasicek paths")
yrs = np.linspace(0.25, 9.5, 60)
zero = -np.log(bond(yrs, r0, a_hat, b_hat, s_hat)) / yrs
ax[1].plot(yrs, zero * 100, color="tab:green")
ax[1].axhline(b_hat * 100, color="grey", ls=":", label="long-run mean")
ax[1].set_xlabel("Maturity (years)"); ax[1].set_ylabel("Zero rate (%)"); ax[1].set_title("Model zero-coupon curve today"); ax[1].legend()
fig.tight_layout(); fig.savefig("reports/pv_scenarios.png", dpi=150)
print("\nsaved reports/pv_scenarios.png")
