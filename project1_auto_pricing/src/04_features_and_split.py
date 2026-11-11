import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CAP = 100_000
con = duckdb.connect("pricing.duckdb")

# Cap each claim, THEN aggregate to one row per policy
con.execute(f"""
CREATE OR REPLACE TABLE sev_agg_cap AS
SELECT s.IDpol, SUM(LEAST(s.ClaimAmount, {CAP})) AS tot_loss_cap
FROM sev s JOIN freq f ON s.IDpol = f.IDpol
GROUP BY s.IDpol
""")

df = con.execute("""
SELECT m.*, COALESCE(c.tot_loss_cap, 0.0) AS tot_loss_cap
FROM model m LEFT JOIN sev_agg_cap c ON m.IDpol = c.IDpol
""").df()

print("rows:", len(df))
print("loss dollars before cap:", round(df.tot_loss.sum(), 2))
print("loss dollars after cap: ", round(df.tot_loss_cap.sum(), 2))
print("share removed by cap:   ", round(1 - df.tot_loss_cap.sum() / df.tot_loss.sum(), 4))

# Features
df["DrivAgeBand"] = pd.cut(df.DrivAge, [17, 25, 35, 45, 55, 65, 100]).astype(str)
df["VehAgeBand"] = pd.cut(df.VehAge, [-1, 2, 5, 10, 100]).astype(str)
df["LogDensity"] = np.log(df.Density)
df["BonusMalus"] = df.BonusMalus.clip(upper=150)
df["freq"] = df.ClaimNb / df.Exposure
df["avg_sev"] = np.where(df.n_sev > 0, df.tot_loss_cap / df.n_sev.clip(lower=1), np.nan)

print("\nunbanded rows (should be 0):",
      int((df.DrivAgeBand == "nan").sum() + (df.VehAgeBand == "nan").sum()))

# Split: fixed seed so results are reproducible
train, test = train_test_split(df, test_size=0.25, random_state=42)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

for name, d in [("train", train), ("test", test)]:
    claims = d.ClaimNb.sum()
    print(f"\n{name}: policies={len(d)}, exposure={d.Exposure.sum():.0f}, claims={claims}")
    print(f"  frequency={claims / d.Exposure.sum():.4f}",
          f" avg severity={d.tot_loss_cap.sum() / d.n_sev.sum():.0f}",
          f" pure premium={d.tot_loss_cap.sum() / d.Exposure.sum():.1f}")
