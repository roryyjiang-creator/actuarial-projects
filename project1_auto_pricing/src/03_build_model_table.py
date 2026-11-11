import duckdb

con = duckdb.connect("pricing.duckdb")

# Aggregate claims to one row per policy BEFORE joining (avoids join fan-out).
# The inner join to freq drops the orphan claims.
con.execute("""
CREATE OR REPLACE TABLE sev_agg AS
SELECT s.IDpol, SUM(s.ClaimAmount) AS tot_loss, COUNT(*) AS n_sev
FROM sev s JOIN freq f ON s.IDpol = f.IDpol
GROUP BY s.IDpol
""")

con.execute("""
CREATE OR REPLACE TABLE model AS
SELECT f.IDpol,
       LEAST(f.Exposure, 1.0) AS Exposure,
       LEAST(COALESCE(a.n_sev, 0), 4) AS ClaimNb,
       COALESCE(a.n_sev, 0) AS n_sev,
       COALESCE(a.tot_loss, 0.0) AS tot_loss,
       f.Area, f.VehPower, f.VehAge, f.DrivAge, f.BonusMalus,
       f.VehBrand, f.VehGas, f.Density, f.Region
FROM freq f LEFT JOIN sev_agg a ON f.IDpol = a.IDpol
""")

def one(q):
    return con.execute(q).fetchone()[0]

print("rows in model:", one("SELECT COUNT(*) FROM model"))
print("rows in freq: ", one("SELECT COUNT(*) FROM freq"))
print("distinct policies in model:", one("SELECT COUNT(DISTINCT IDpol) FROM model"))

total_sev = one("SELECT SUM(ClaimAmount) FROM sev")
orphan = one("""SELECT COALESCE(SUM(s.ClaimAmount), 0) FROM sev s
                LEFT JOIN freq f ON s.IDpol = f.IDpol WHERE f.IDpol IS NULL""")
in_model = one("SELECT SUM(tot_loss) FROM model")
print("\nReconciliation of loss dollars")
print("total in claims table:   ", round(total_sev, 2))
print("orphan claims (dropped): ", round(orphan, 2))
print("total in model table:    ", round(in_model, 2))
print("difference (should be 0):", round(total_sev - orphan - in_model, 2))

print("\nClaim amount distribution:")
print(con.execute("""
SELECT MIN(ClaimAmount) AS min, MEDIAN(ClaimAmount) AS median,
       QUANTILE_CONT(ClaimAmount, 0.99) AS p99, MAX(ClaimAmount) AS max,
       SUM(CASE WHEN ClaimAmount > 100000 THEN 1 ELSE 0 END) AS over_100k
FROM sev
""").df())

print("\nClaim counts in model:")
print(con.execute("SELECT ClaimNb, COUNT(*) AS policies FROM model GROUP BY ClaimNb ORDER BY ClaimNb").df())
