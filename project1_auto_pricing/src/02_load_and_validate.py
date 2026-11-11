import duckdb

con = duckdb.connect("pricing.duckdb")

# Load policies, stripping stray quote characters from text columns
con.execute("""
CREATE OR REPLACE TABLE freq AS
SELECT IDpol, ClaimNb, Exposure,
       REPLACE(Area, chr(39), '') AS Area,
       VehPower, VehAge, DrivAge, BonusMalus,
       REPLACE(VehBrand, chr(39), '') AS VehBrand,
       REPLACE(VehGas, chr(39), '') AS VehGas,
       Density,
       REPLACE(Region, chr(39), '') AS Region
FROM read_csv_auto('data/freq.csv')
""")

# Load claims
con.execute("""
CREATE OR REPLACE TABLE sev AS
SELECT CAST(IDpol AS BIGINT) AS IDpol, ClaimAmount
FROM read_csv_auto('data/sev.csv')
""")

checks = {
    "policy rows": "SELECT COUNT(*) FROM freq",
    "claim rows": "SELECT COUNT(*) FROM sev",
    "duplicate policy IDs": "SELECT COUNT(*) FROM (SELECT IDpol FROM freq GROUP BY IDpol HAVING COUNT(*) > 1)",
    "exposure <= 0 or > 1": "SELECT COUNT(*) FROM freq WHERE Exposure <= 0 OR Exposure > 1",
    "ClaimNb > 4": "SELECT COUNT(*) FROM freq WHERE ClaimNb > 4",
    "claims with no matching policy": "SELECT COUNT(*) FROM sev s LEFT JOIN freq f ON s.IDpol = f.IDpol WHERE f.IDpol IS NULL",
    "policies with ClaimNb>0 but no claim record": "SELECT COUNT(*) FROM freq WHERE ClaimNb > 0 AND IDpol NOT IN (SELECT IDpol FROM sev)",
    "policies with claim record but ClaimNb=0": "SELECT COUNT(DISTINCT s.IDpol) FROM sev s JOIN freq f ON s.IDpol = f.IDpol WHERE f.ClaimNb = 0",
    "non-positive claim amounts": "SELECT COUNT(*) FROM sev WHERE ClaimAmount <= 0",
}

for name, query in checks.items():
    print(f"{name}: {con.execute(query).fetchone()[0]}")

print("\nVehGas values:", con.execute("SELECT DISTINCT VehGas FROM freq").fetchall())
print("\nClaimNb vs rows in claims table:")
print(con.execute("""
SELECT f.ClaimNb, COUNT(*) AS policies, ROUND(AVG(COALESCE(s.n, 0)), 2) AS avg_claim_rows
FROM freq f LEFT JOIN (SELECT IDpol, COUNT(*) AS n FROM sev GROUP BY IDpol) s ON f.IDpol = s.IDpol
GROUP BY f.ClaimNb ORDER BY f.ClaimNb
""").df())
