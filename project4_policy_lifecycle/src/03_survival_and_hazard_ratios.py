"""Kaplan-Meier survival curves and hazard ratios for the rating factor.

(A) Nonparametric Kaplan-Meier curves of time to lapse by group, against the Markov model.
(B) Poisson regression for hazards: log(rate) = b0 + b1*high_risk (+ b2*in_claim for lapse),
    with the time in each state as exposure. exp(b) is a hazard ratio, the survival analogue of a
    rating relativity. Run from the project4_policy_lifecycle folder.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

T = 10.0
h = pd.read_csv("data/histories.csv")
h["dur"] = h.end - h.start
pd.set_option("display.width", 200)

# ---------- (A) Kaplan-Meier ----------
pol = h.groupby("policy_id").agg(g=("high_risk", "first"), time=("end", "max"),
                                 lapsed=("next_state", lambda s: (s == 2).any()))

def km(times, events):
    """Kaplan-Meier estimate at each observation (continuous times, no ties assumed)."""
    order = np.argsort(times, kind="stable")
    t, e = times[order], events[order].astype(float)
    at_risk = len(t) - np.arange(len(t))
    return t, np.cumprod(1 - e / at_risk)

def km_at(t, S, x):
    i = np.searchsorted(t, x, side="right") - 1
    return 1.0 if i < 0 else S[i]

# rates from the data (same MLE as script 02)
def mle_Q(g):
    hg = h[h.high_risk == g]
    r = {}
    for (i, j) in ((0, 1), (0, 2), (1, 0), (1, 2)):
        r[(i, j)] = ((hg.state == i) & (hg.next_state == j)).sum() / hg.loc[hg.state == i, "dur"].sum()
    return np.array([[-(r[0, 1] + r[0, 2]), r[0, 1]], [r[1, 0], -(r[1, 0] + r[1, 2])]])

curves, rows = {}, []
for g, label in ((0, "low risk"), (1, "high risk")):
    pg = pol[pol.g == g]
    t, S = km(pg.time.to_numpy(), pg.lapsed.to_numpy())
    Q = mle_Q(g)
    curves[g] = (t, S, Q)
    for x in (1, 2, 5, 10):
        rows.append({"group": label, "years": x, "Kaplan-Meier": km_at(t, S, x),
                     "Markov model": (expm(Q * x) @ np.ones(2))[0]})
print("(A) SHARE OF POLICIES STILL IN FORCE")
print(pd.DataFrame(rows).round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 4.5))
grid = np.linspace(0, T, 200)
for g, label, color in ((0, "low risk", "tab:blue"), (1, "high risk", "tab:red")):
    t, S, Q = curves[g]
    ax.step(np.r_[0, t], np.r_[1, S], where="post", color=color, label=f"{label}: Kaplan-Meier")
    ax.plot(grid, [(expm(Q * x) @ np.ones(2))[0] for x in grid], "--", color="black", lw=1,
            label="Markov model" if g == 0 else None)
ax.set_xlabel("Years since policy start"); ax.set_ylabel("Share of policies still in force")
ax.set_title("Retention: Kaplan-Meier vs fitted Markov chain"); ax.legend(); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig("reports/retention_curves.png", dpi=150)

# ---------- (B) hazard ratios by Poisson regression ----------
def poisson_fit(X, y, expo, iters=50):
    """Poisson GLM with log link and log(exposure) offset, fitted by Newton-Raphson."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.sum() / expo.sum())
    for _ in range(iters):
        mu = expo * np.exp(X @ beta)
        step = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    mu = expo * np.exp(X @ beta)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * mu[:, None]))))
    return beta, se

def table(names, beta, se, truth):
    return pd.DataFrame({"effect": names, "hazard ratio": np.exp(beta[1:]),
                         "95% low": np.exp(beta[1:] - 1.96 * se[1:]),
                         "95% high": np.exp(beta[1:] + 1.96 * se[1:]), "true": truth})

# lapse hazard: cells defined by (high_risk, state); events = lapses, exposure = time in state
cells = h.groupby(["high_risk", "state"]).agg(events=("next_state", lambda s: (s == 2).sum()),
                                               exposure=("dur", "sum")).reset_index()
Xl = np.column_stack([np.ones(len(cells)), cells.high_risk, cells.state])
bl, sl = poisson_fit(Xl, cells.events.to_numpy(float), cells.exposure.to_numpy())
print("\n(B) LAPSE HAZARD RATIOS (time-varying 'open claim' as a covariate)")
print(table(["high risk vs low risk", "in claim vs active"], bl, sl, [1.4, 0.40 / 0.15]).round(3).to_string(index=False))

# claim hazard (Active -> Claim): only time in Active is at risk
a = h[h.state == 0].groupby("high_risk").agg(events=("next_state", lambda s: (s == 1).sum()),
                                              exposure=("dur", "sum")).reset_index()
Xc = np.column_stack([np.ones(len(a)), a.high_risk])
bc, sc = poisson_fit(Xc, a.events.to_numpy(float), a.exposure.to_numpy())
print("\nCLAIM HAZARD RATIO (Active -> Claim)")
print(table(["high risk vs low risk"], bc, sc, [2.0]).round(3).to_string(index=False))

try:  # cross-check against statsmodels if it is installed
    import statsmodels.api as sm
    m = sm.GLM(cells.events, Xl, family=sm.families.Poisson(), offset=np.log(cells.exposure)).fit()
    print("\nstatsmodels check, lapse hazard ratios:", np.exp(m.params[1:]).round(3).tolist())
except ImportError:
    pass
print("\nsaved reports/retention_curves.png")
