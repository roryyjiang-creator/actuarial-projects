"""Present value of a policy's premiums and claims, using the fitted Markov chain.

Premium P per year is paid continuously while the policy is in force (Active or Claim).
A claim costs C, paid when the policy moves Active -> Claim. Acquisition cost A0 is paid at start.
With force of interest d and transient generator Q (Active, Claim; lapse is absorbing):
    PV of time in states over [0, T] = alpha (dI - Q)^-1 (I - exp((Q - dI)T)) 1
Run from the project4_policy_lifecycle folder.
"""
import numpy as np
import pandas as pd
from scipy.linalg import expm

T = 10.0
P, C, A0 = 1000.0, 5000.0, 600.0       # assumptions: premium/yr, cost per claim, acquisition cost
D = 0.04                                # base force of interest
pd.set_option("display.width", 200)

h = pd.read_csv("data/histories.csv")
h["dur"] = h.end - h.start

def mle_Q(g):
    hg = h[h.high_risk == g]
    r = {}
    for (i, j) in ((0, 1), (0, 2), (1, 0), (1, 2)):
        r[(i, j)] = ((hg.state == i) & (hg.next_state == j)).sum() / hg.loc[hg.state == i, "dur"].sum()
    return np.array([[-(r[0, 1] + r[0, 2]), r[0, 1]], [r[1, 0], -(r[1, 0] + r[1, 2])]])

alpha = np.array([1.0, 0.0])

def discounted_time(Q, d, T=T):
    """Vector of expected discounted time in each state over [0, T], starting Active."""
    M = d * np.eye(2) - Q
    return alpha @ np.linalg.solve(M, np.eye(2) - expm((Q - d * np.eye(2)) * T))

def pv(Q, d, premium=P, claim=C, acq=A0, T=T):
    tt = discounted_time(Q, d, T)
    prem = premium * tt.sum()
    clm = claim * Q[0, 1] * tt[0]       # claims arrive at rate q_AC while Active
    return prem, clm, prem - clm - acq

def scaled(Q, k):
    """Multiply the lapse rates by k (k < 1 = better retention)."""
    Q2 = Q.copy()
    l_a, l_c = -Q[0].sum(), -Q[1].sum()    # (row sums are minus the lapse rate)
    Q2[0, 0] -= (k - 1) * l_a
    Q2[1, 1] -= (k - 1) * l_c
    return Q2

Q = {g: mle_Q(g) for g in (0, 1)}
label = {0: "low risk", 1: "high risk"}

# ---- 1. base case ----
rows = []
for g in (0, 1):
    p, c, m = pv(Q[g], D)
    rows.append({"group": label[g], "PV premium": p, "PV claims": c, "acquisition": A0, "PV profit": m,
                 "profit / premium": m / p})
print(f"1. BASE CASE: premium {P:,.0f}/yr, claim cost {C:,.0f}, acquisition {A0:,.0f}, interest {D:.0%}, horizon {T:.0f} years")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- 2. check against the simulated histories ----
def pv_data(g, d=D, premium=P, claim=C, acq=A0):
    hg = h[h.high_risk == g]
    n = hg.policy_id.nunique()
    prem = premium * ((np.exp(-d * hg.start) - np.exp(-d * hg.end)) / d).sum() / n
    clm = claim * ((hg.state == 0) & (hg.next_state == 1)).mul(np.exp(-d * hg.end)).sum() / n
    return prem, clm, prem - clm - acq

rows = []
for g in (0, 1):
    pm, cm, mm = pv(Q[g], D)
    pd_, cd, md = pv_data(g)
    rows.append({"group": label[g], "model premium": pm, "data premium": pd_, "model claims": cm,
                 "data claims": cd, "model profit": mm, "data profit": md})
print("\n2. CHECK: model vs direct average over the simulated policies")
print(pd.DataFrame(rows).round(1).to_string(index=False))

# ---- 3. sensitivity to interest ----
rows = []
for d in (0.001, 0.02, 0.04, 0.06, 0.08):
    r = {"interest": f"{d:.1%}" if d < 0.01 else f"{d:.0%}"}
    for g in (0, 1):
        r[f"{label[g]} profit"] = pv(Q[g], d)[2]
    rows.append(r)
print("\n3. SENSITIVITY: PV profit per policy vs interest rate")
print(pd.DataFrame(rows).round(0).to_string(index=False))

# ---- 4. sensitivity to retention ----
rows = []
for k in (0.5, 0.75, 1.0, 1.25, 1.5):
    r = {"lapse rates x": k}
    for g in (0, 1):
        r[f"{label[g]} profit"] = pv(scaled(Q[g], k), D)[2]
        r[f"{label[g]} yrs in force"] = discounted_time(scaled(Q[g], k), 1e-9, 200).sum()
    rows.append(r)
print("\n4. SENSITIVITY: PV profit per policy vs lapse rates (yrs in force = expected, no horizon limit)")
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---- 5. breakeven premium, and what a bad pricing assumption costs ----
rows = []
for g in (0, 1):
    p, c, m = pv(Q[g], D)
    pol_years = discounted_time(Q[g], D).sum()
    be = (c + A0) / pol_years
    rows.append({"group": label[g], "breakeven premium/yr": be, "current premium": P,
                 "margin at P": P / be - 1})
print("\n5. BREAKEVEN PREMIUM (PV profit = 0)")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# mispricing: price high-risk using low-risk rates
p_low, c_low, _ = pv(Q[0], D)
be_low = (c_low + A0) / discounted_time(Q[0], D).sum()
_, _, m_hi = pv(Q[1], D, premium=be_low)
print(f"\nIf high-risk policies are charged the low-risk breakeven premium ({be_low:,.0f}/yr), "
      f"PV profit per high-risk policy = {m_hi:,.0f}")
