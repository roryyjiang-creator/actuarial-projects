"""Validation checks and a first look at the cost distribution. Run before any modeling."""
import numpy as np
import pandas as pd

df = pd.read_csv("data/insurance.csv")
expected = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
assert list(df.columns) == expected, f"unexpected columns: {list(df.columns)}"

checks = {
    "rows": len(df),
    "exact duplicate rows": int(df.duplicated().sum()),
    "missing values": int(df.isna().sum().sum()),
    "age outside 18-100": int(((df.age < 18) | (df.age > 100)).sum()),
    "bmi outside 10-70": int(((df.bmi < 10) | (df.bmi > 70)).sum()),
    "negative or zero charges": int((df.charges <= 0).sum()),
    "children outside 0-10": int(((df.children < 0) | (df.children > 10)).sum()),
}
print("VALIDATION CHECKS")
for k, v in checks.items():
    print(f"  {k}: {v}")

print("\nCATEGORY LEVELS")
for c in ("sex", "smoker", "region"):
    print(f"  {c}: {df[c].value_counts().to_dict()}")

dups = df[df.duplicated(keep=False)]
if len(dups):
    print("\nDUPLICATED ROWS (all copies):")
    print(dups.to_string())

print("\nCHARGES: SKEW AND TAIL")
q = df.charges.quantile([0.5, 0.9, 0.99])
print(f"  mean {df.charges.mean():,.0f}   median {q[0.5]:,.0f}   mean/median {df.charges.mean() / q[0.5]:.2f}")
print(f"  90th pct {q[0.9]:,.0f}   99th pct {q[0.99]:,.0f}   max {df.charges.max():,.0f}")
top = df.nlargest(int(round(0.1 * len(df))), "charges")
print(f"  top 10% of people account for {top.charges.sum() / df.charges.sum():.1%} of total charges")

print("\nBY SMOKER")
g = df.groupby("smoker").charges.agg(["count", "mean", "median"])
g["share of people"] = g["count"] / len(df)
g["share of charges"] = df.groupby("smoker").charges.sum() / df.charges.sum()
print(g.round(3).to_string())
