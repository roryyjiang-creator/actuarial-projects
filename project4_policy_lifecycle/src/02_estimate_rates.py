"""Estimate transition rates by maximum likelihood and derive Markov chain quantities.

For a continuous-time Markov chain observed with censoring, the MLE of the rate i -> j is
(number of i -> j transitions) / (total time spent in state i). Run from the project folder.
"""
import numpy as np
import pandas as pd
from scipy.linalg import expm

T = 10.0
TRUE = {  # same values as script 01
    0: {"A->C": 0.08, "A->L": 0.15, "C->A": 6.0, "C->L": 0.40},
    1: {"A->C": 0.16, "A->L": 0.21, "C->A": 6.0, "C->L": 0.56}}
KEY = {(0, 1): "A->C", (0, 2): "A->L", (1, 0): "C->A", (1, 2): "C->L"}

h = pd.read_csv("data/histories.csv")
h["dur"] = h.end - h.start

def transient_Q(r):
    """Generator on the transient states (Active, Claim); lapse is absorbing and dropped."""
    return np.array([[-(r["A->C"] + r["A->L"]), r["A->C"]],
                     [r["C->A"], -(r["C->A"] + r["C->L"])]])

est_rows, Qhat, Qtrue = [], {}, {}
for g in (0, 1):
    hg = h[h.high_risk == g]
    r_hat = {}
    for (i, j), name in KEY.items():
        n = int(((hg.state == i) & (hg.next_state == j)).sum())
        exposure = hg.loc[hg.state == i, "dur"].sum()
        rate = n / exposure
        r_hat[name] = rate
        est_rows.append({"group": "high risk" if g else "low risk", "transition": name,
                         "events": n, "years in state": exposure, "true rate": TRUE[g][name],
                         "estimate": rate, "95% low": rate * (1 - 1.96 / np.sqrt(n)),
                         "95% high": rate * (1 + 1.96 / np.sqrt(n))})
    Qhat[g], Qtrue[g] = transient_Q(r_hat), transient_Q(TRUE[g])

est = pd.DataFrame(est_rows)
pd.set_option("display.width", 200)
print("MAXIMUM LIKELIHOOD RATES (per year)")
print(est.round(3).to_string(index=False))
est.to_csv("reports/estimated_rates.csv", index=False)

print("\nEXPECTED YEARS IN EACH STATE BEFORE LAPSE (no time limit): -alpha * Q^-1, start Active")
alpha = np.array([1.0, 0.0])
rows = []
for g, label in ((0, "low risk"), (1, "high risk")):
    for src, Q in (("true", Qtrue[g]), ("estimated", Qhat[g])):
        t = -alpha @ np.linalg.inv(Q)
        rows.append({"group": label, "rates": src, "years Active": t[0],
                     "years in Claim": t[1], "expected years in force": t.sum()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print(f"\nCHECK AGAINST DATA: expected years in each state within {T:.0f} years")
N = h.policy_id.nunique()
rows = []
for g, label in ((0, "low risk"), (1, "high risk")):
    hg = h[h.high_risk == g]
    n_g = hg.policy_id.nunique()
    # integral_0^T exp(Q t) dt = Q^-1 (exp(QT) - I)
    M = np.linalg.inv(Qhat[g]) @ (expm(Qhat[g] * T) - np.eye(2))
    model = alpha @ M
    emp = [hg.loc[hg.state == s, "dur"].sum() / n_g for s in (0, 1)]
    rows.append({"group": label, "model Active": model[0], "data Active": emp[0],
                 "model Claim": model[1], "data Claim": emp[1]})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nPROBABILITY STILL IN FORCE (model vs data)")
last = h.groupby("policy_id").agg(g=("high_risk", "first"), exit=("end", "max"),
                                  lapsed=("next_state", lambda s: (s == 2).any()))
rows = []
for g, label in ((0, "low risk"), (1, "high risk")):
    lg = last[last.g == g]
    for t in (1, 5, 10):
        p = (expm(Qhat[g] * t) @ np.ones(2))[0]
        data = 1 - ((lg.lapsed) & (lg.exit <= t)).mean()
        rows.append({"group": label, "years": t, "model": p, "data": data})
print(pd.DataFrame(rows).round(3).to_string(index=False))
