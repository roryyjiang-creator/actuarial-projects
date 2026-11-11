"""Simulate policyholder histories from a continuous-time Markov chain.

States: 0 = Active, 1 = Claim (open claim, policy still in force), 2 = Lapsed (absorbing).
Rates are per year. A binary rating factor 'high_risk' multiplies the claim rate by 2.0
and both lapse rates by 1.4. Observation stops at T = 10 years (censoring).
Run from the project4_policy_lifecycle folder.
"""
import numpy as np
import pandas as pd

SEED, N_POLICIES, T, P_HIGH = 2026, 20_000, 10.0, 0.30
BASE = {"claim": 0.08, "lapse_active": 0.15, "recover": 6.0, "lapse_claim": 0.40}
MULT = {"claim": 2.0, "lapse_active": 1.4, "recover": 1.0, "lapse_claim": 1.4}

def rates(g):
    m = lambda k: BASE[k] * (MULT[k] if g else 1.0)
    return {0: {1: m("claim"), 2: m("lapse_active")},   # Active -> Claim, Active -> Lapsed
            1: {0: m("recover"), 2: m("lapse_claim")}}  # Claim -> Active, Claim -> Lapsed

RATES = {0: rates(0), 1: rates(1)}
rng = np.random.default_rng(SEED)

def simulate_policy(pid, g):
    state, t, rows = 0, 0.0, []
    while True:
        targets, rs = zip(*RATES[g][state].items())
        total = sum(rs)
        end = t + rng.exponential(1 / total)
        if end >= T:
            rows.append((pid, g, state, t, T, -1))          # censored at T
            return rows
        nxt = targets[rng.choice(len(rs), p=np.array(rs) / total)]
        rows.append((pid, g, state, t, end, nxt))
        if nxt == 2:
            return rows
        state, t = nxt, end

rows = []
for pid in range(N_POLICIES):
    rows += simulate_policy(pid, int(rng.random() < P_HIGH))
h = pd.DataFrame(rows, columns=["policy_id", "high_risk", "state", "start", "end", "next_state"])
h.to_csv("data/histories.csv", index=False)

pol = h.groupby("policy_id").agg(high_risk=("high_risk", "first"), exit=("end", "max"),
                                 lapsed=("next_state", lambda s: (s == 2).any()))
print(f"{N_POLICIES:,} policies, {len(h):,} spells, observed for up to {T:.0f} years")
print(f"high-risk share: {pol.high_risk.mean():.3f}")
print("transitions observed:")
print(h[h.next_state >= 0].groupby(["high_risk", "state", "next_state"]).size().unstack(fill_value=0))
print(f"\nlapsed within {T:.0f} years: {pol.lapsed.mean():.1%} "
      f"(high risk {pol[pol.high_risk == 1].lapsed.mean():.1%}, "
      f"low risk {pol[pol.high_risk == 0].lapsed.mean():.1%})")
print(f"claims per policy: {(h.next_state.eq(0) | (h.state.eq(1))).groupby(h.policy_id).sum().mean():.3f}")
print("saved data/histories.csv")
