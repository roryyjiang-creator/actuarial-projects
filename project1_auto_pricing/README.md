# Project 1: Auto Insurance Frequency-Severity Pricing

Goal: build a pure premium model (frequency x severity) on the French motor
third-party liability data (freMTPL2freq and freMTPL2sev from OpenML).

## Reproducing the data

Run `python src/01_download_data.py`, which downloads the data into `data/`.
The data files are not committed to the repository because of their size.

## Data validation

The policy table has 678,013 rows and the claims table has 26,639 rows.
I checked the data before modeling and made these decisions:

| Check | Result | Decision |
|---|---|---|
| Duplicate policy IDs | 0 | None needed |
| Exposure outside (0, 1] | 1,224 policies | Clip exposure at 1 |
| Claim count above 4 | 9 policies | Cap claim count at 4 |
| Claims with no matching policy | 195 claims | Drop them; they have no exposure or rating variables |
| Claim count above 0 but no claim record | 9,116 policies | Define claim count from the claims table so frequency and severity are consistent |
| Claim record but claim count 0 | 0 | None needed |
| Non-positive claim amounts | 0 | None needed |
| VehGas values contained stray quote characters | All rows | Stripped the quotes when loading |

Claims were aggregated to one row per policy before joining to the policy table,
to avoid double-counting exposure. Loss dollars were reconciled between the
claims table and the model table.