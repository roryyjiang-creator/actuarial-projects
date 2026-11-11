"""Build excel/auto_pricing_calculator.xlsx from the saved GLM relativities."""
import re
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.workbook.defined_name import DefinedName

LOAD, N_CHECK, SEED = 1.228, 20_000, 42
PAT = re.compile(r"^C\((\w+),.*?reference='([^']*)'\)\)\[T\.(.*)\]$")

AGE_UP, AGE_LAB = [25, 35, 45, 55, 65, 100], ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
VA_UP, VA_LAB = [2, 5, 10, 100], ["0-2", "3-5", "6-10", "11+"]
BM_UP = [50, 60, 75, 90, 100, 110, 125, 150]
BM_LAB = ["<=50", "51-60", "61-75", "76-90", "91-100", "101-110", "111-125", "126-150"]

# ---------- read relativities ----------
rel_rows, BASE = [], {}
def read_model(path, model):
    df = pd.read_csv(path, index_col=0)
    levels = {}
    for term, r in df.iterrows():
        m = PAT.match(term)
        if term == "Intercept":
            rel_rows.append((model, "Intercept", "base", r.relativity, r.ci_low, r.ci_high))
        elif m:
            var, ref, lvl = m.groups()
            BASE.setdefault((model, var), ref)
            levels.setdefault(var, []).append((lvl, r.relativity, r.ci_low, r.ci_high))
        else:
            rel_rows.append((model, term, "per unit", r.relativity, r.ci_low, r.ci_high))
    for var, lst in levels.items():
        rel_rows.append((model, var, BASE[(model, var)], 1.0, 1.0, 1.0))
        for lvl, a, b, c in lst:
            rel_rows.append((model, var, lvl, a, b, c))
    return df

fq = read_model("reports/frequency_v2_relativities.csv", "Freq")
sv = read_model("reports/severity_relativities.csv", "Sev")
D = {(m, v, l): r for m, v, l, r, _, _ in rel_rows}
VARS = {m: {v for (mm, v, _) in D if mm == m} for m in ("Freq", "Sev")}
print("Frequency terms:", sorted(VARS["Freq"]))
print("Severity terms: ", sorted(VARS["Sev"]))

allvals = pd.concat([fq.relativity, sv.relativity])
if ((allvals * 1000 - (allvals * 1000).round()).abs() < 1e-6).all():
    print("\nWARNING: relativities look rounded to 3 decimals. Continuous terms such as "
          "BonusMalus are raised to large powers, so rounding can move results by several "
          "percent. Save with more decimals (see instructions) and rerun.\n")

def rel(m, v, l):
    return D[(m, v, l)]

# ---------- workbook ----------
wb = Workbook()
cal = wb.active; cal.title = "Calculator"
rec = wb.create_sheet("Reconciliation")
chk = wb.create_sheet("Check")
rs = wb.create_sheet("Rel")
ls = wb.create_sheet("Lists")

HDR = Font(bold=True, color="FFFFFF"); HFILL = PatternFill("solid", fgColor="305496")
INP = PatternFill("solid", fgColor="FFF2CC"); BLUE = Font(color="0000FF")
def header(ws, row, vals, col=1):
    for j, v in enumerate(vals):
        c = ws.cell(row=row, column=col + j, value=v)
        c.font, c.fill = HDR, HFILL
        c.alignment = Alignment(horizontal="center", wrap_text=True)

# Rel sheet
header(rs, 1, ["Key", "Model", "Variable", "Level", "Relativity", "CI low", "CI high"])
for i, (m, v, l, r, lo, hi) in enumerate(rel_rows, start=2):
    rs.cell(row=i, column=1, value=f'=B{i}&"|"&C{i}&"|"&D{i}')
    for j, val in enumerate([m, v, str(l), float(r), float(lo), float(hi)], start=2):
        rs.cell(row=i, column=j, value=val)
    for j in (5, 6, 7):
        rs.cell(row=i, column=j).number_format = "0.000000"
last = len(rel_rows) + 1
for name, col in (("RelKey", "A"), ("RelVal", "E")):
    wb.defined_names[name] = DefinedName(name, attr_text=f"Rel!${col}$2:${col}${last}")
for col, w in zip("ABCDEFG", (34, 8, 20, 14, 12, 10, 10)):
    rs.column_dimensions[col].width = w
rs.freeze_panes = "A2"

def XL(key_expr):
    return f"_xlfn.XLOOKUP({key_expr},RelKey,RelVal)"

# Lists sheet
def levels_of(var):
    out = []
    for m in ("Freq", "Sev"):
        for (mm, v, l) in D:
            if mm == m and v == var and l not in out:
                out.append(l)
    return out
header(ls, 1, ["VehGas", "VehBrand", "Region", "Area"])
list_ranges = {}
for j, var in enumerate(["VehGas", "VehBrand", "Region", "Area"], start=1):
    lv = levels_of(var)
    for i, l in enumerate(lv, start=2):
        ls.cell(row=i, column=j, value=l)
    L = chr(64 + j)
    list_ranges[var] = f"=Lists!${L}$2:${L}${max(len(lv), 1) + 1}"
header(ls, 1, ["AgeUpper", "AgeBand", "VehAgeUpper", "VehAgeBand", "BMUpper", "BMBand"], col=6)
for j, (up, lab) in enumerate([(AGE_UP, AGE_LAB), (VA_UP, VA_LAB), (BM_UP, BM_LAB)]):
    for i, (u, l) in enumerate(zip(up, lab), start=2):
        ls.cell(row=i, column=6 + 2 * j, value=u)
        ls.cell(row=i, column=7 + 2 * j, value=l)
for name, col, n in (("AgeUpper", "F", 6), ("AgeLabel", "G", 6), ("VAUpper", "H", 4), ("VALabel", "I", 4),
                     ("BMUpper", "J", 8), ("BMLabel", "K", 8)):
    wb.defined_names[name] = DefinedName(name, attr_text=f"Lists!${col}$2:${col}${n + 1}")
for c in "ABCDFGHIJK":
    ls.column_dimensions[c].width = 13

# Calculator sheet
cal["A1"] = "Auto pricing calculator (frequency x severity GLMs)"; cal["A1"].font = Font(bold=True, size=14)
cal["A2"] = "Yellow cells are inputs. Relativities come from the Rel sheet via XLOOKUP."
header(cal, 3, ["Rating factor", "Input", "Level used", "Frequency relativity", "Severity relativity"])
cal["A4"] = "Intercept (base rate)"
cal["D4"] = "=" + XL('"Freq|Intercept|base"'); cal["E4"] = "=" + XL('"Sev|Intercept|base"')

def base_of(var):
    for m in ("Freq", "Sev"):
        if (m, var) in BASE:
            return BASE[(m, var)]
def num_default(var, fallback):
    try:
        return int(float(base_of(var)))
    except Exception:
        return fallback

def cat_formula(model, var, cell):
    return XL(f'"{model}|{var}|"&{cell}') if var in VARS[model] else 1

rows = [
    ("Driver age (years)", 50, 'XL_BAND', "DrivAgeBand", f"=_xlfn.XLOOKUP(B5,AgeUpper,AgeLabel,\"?\",1)"),
    ("Vehicle age (years)", 12, 'XL_BAND', "VehAgeBand", f"=_xlfn.XLOOKUP(B6,VAUpper,VALabel,\"?\",1)"),
    ("Vehicle power", num_default("VehPowerC", 6), 'TEXT', "VehPowerC", '=TEXT(MIN(MAX(B7,4),10),"0")'),
    ("Fuel", base_of("VehGas"), 'CAT', "VehGas", "=B8"),
    ("Vehicle brand", base_of("VehBrand"), 'CAT', "VehBrand", "=B9"),
    ("Region", base_of("Region"), 'CAT', "Region", "=B10"),
    ("Area", base_of("Area"), 'CAT', "Area", "=B11"),
]
r = 5
for label, default, kind, var, cform in rows:
    cal.cell(row=r, column=1, value=label)
    c = cal.cell(row=r, column=2, value=default); c.fill, c.font = INP, BLUE
    cal.cell(row=r, column=3, value=cform)
    for col, model in ((4, "Freq"), (5, "Sev")):
        f = cat_formula(model, var, f"C{r}")
        cal.cell(row=r, column=col, value=f if f == 1 else "=" + f)
    r += 1
# density (continuous) row 12
cal.cell(row=12, column=1, value="Density (people per km2)")
c = cal.cell(row=12, column=2, value=100); c.fill, c.font = INP, BLUE
cal["C12"] = "=LN(B12)"
for col, model in ((4, "Freq"), (5, "Sev")):
    cal.cell(row=12, column=col,
             value=(f'={XL(chr(34) + model + "|LogDensity|per unit" + chr(34))}^C12') if "LogDensity" in VARS[model] else 1)
# bonus-malus row 13
cal.cell(row=13, column=1, value="Bonus-malus")
c = cal.cell(row=13, column=2, value=50); c.fill, c.font = INP, BLUE
cal["C13"] = '=_xlfn.XLOOKUP(MIN(B13,150),BMUpper,BMLabel,"?",1)'
cal["D13"] = ("=" + XL('"Freq|BMBand|"&C13')) if "BMBand" in VARS["Freq"] else (
    f'={XL(chr(34) + "Freq|BonusMalus|per unit" + chr(34))}^MIN(B13,150)' if "BonusMalus" in VARS["Freq"] else 1)
cal["E13"] = f'={XL(chr(34) + "Sev|BonusMalus|per unit" + chr(34))}^MIN(B13,150)' if "BonusMalus" in VARS["Sev"] else (
    ("=" + XL('"Sev|BMBand|"&C13')) if "BMBand" in VARS["Sev"] else 1)

cal["A15"] = "Predicted claim frequency (per exposure year)"; cal["B15"] = "=PRODUCT(D4:D13)"
cal["A16"] = "Predicted severity per claim (capped at 100,000)"; cal["B16"] = "=PRODUCT(E4:E13)"
cal["A17"] = "Large-loss load"; cal["B17"] = LOAD; cal["B17"].fill, cal["B17"].font = INP, BLUE
cal["A18"] = "Exposure (years)"; cal["B18"] = 1; cal["B18"].fill, cal["B18"].font = INP, BLUE
cal["A19"] = "Expected loss cost (pure premium)"; cal["B19"] = "=B15*B16*B17*B18"
cal["A19"].font = cal["B19"].font = Font(bold=True)
cal["B15"].number_format = "0.0000"; cal["B16"].number_format = "#,##0.00"
cal["B17"].number_format = "0.000"; cal["B19"].number_format = "#,##0.00"
for rr in range(4, 14):
    for cc in ("D", "E"):
        cal[f"{cc}{rr}"].number_format = "0.0000"
cal["A21"] = ("Pure premium = frequency x severity x large-loss load x exposure. Excludes expenses, "
              "profit and trend.")
for col, w in zip("ABCDE", (46, 14, 14, 20, 20)):
    cal.column_dimensions[col].width = w
for var, cell in (("VehGas", "B8"), ("VehBrand", "B9"), ("Region", "B10"), ("Area", "B11")):
    dv = DataValidation(type="list", formula1=list_ranges[var], allow_blank=False)
    cal.add_data_validation(dv); dv.add(cell)
for cell, lo, hi in (("B5", 18, 100), ("B6", 0, 100), ("B13", 50, 230)):
    dv = DataValidation(type="whole", operator="between", formula1=str(lo), formula2=str(hi))
    cal.add_data_validation(dv); dv.add(cell)

# ---------- check sample ----------
test = pd.read_csv("data/test.csv")
need = ["ClaimNb", "Exposure", "DrivAge", "VehAge", "VehPower", "VehGas", "VehBrand",
        "Region", "Area", "Density", "BonusMalus"]
missing = [c for c in need if c not in test.columns]
if missing:
    raise SystemExit(f"data/test.csv is missing columns {missing}. Columns present: {list(test.columns)}")
s = test.sample(n=min(N_CHECK, len(test)), random_state=SEED).reset_index(drop=True)[need].copy()
for c in ("VehGas", "VehBrand", "Region", "Area"):
    s[c] = s[c].astype(str).str.strip("'\" ")

header(chk, 1, need + ["DrivAgeBand", "VehAgeBand", "VehPowerC", "BMBand", "PredFreq", "PredClaims"])
n = len(s)
fint = XL('"Freq|Intercept|base"')
for i, row in enumerate(s.itertuples(index=False), start=2):
    for j, v in enumerate(row, start=1):
        chk.cell(row=i, column=j, value=v.item() if hasattr(v, "item") else v)
    chk.cell(row=i, column=12, value=f'=_xlfn.XLOOKUP(C{i},AgeUpper,AgeLabel,"?",1)')
    chk.cell(row=i, column=13, value=f'=_xlfn.XLOOKUP(D{i},VAUpper,VALabel,"?",1)')
    chk.cell(row=i, column=14, value=f'=TEXT(MIN(MAX(E{i},4),10),"0")')
    chk.cell(row=i, column=15, value=f'=_xlfn.XLOOKUP(MIN(K{i},150),BMUpper,BMLabel,"?",1)')
    parts = [fint]
    for var, cell in (("DrivAgeBand", f"L{i}"), ("VehAgeBand", f"M{i}"), ("VehPowerC", f"N{i}"),
                      ("VehGas", f"F{i}"), ("VehBrand", f"G{i}"), ("Region", f"H{i}"),
                      ("Area", f"I{i}"), ("BMBand", f"O{i}")):
        if var in VARS["Freq"]:
            parts.append(XL(f'"Freq|{var}|"&{cell}'))
    if "LogDensity" in VARS["Freq"]:
        parts.append(f'{XL(chr(34) + "Freq|LogDensity|per unit" + chr(34))}^LN(J{i})')
    chk.cell(row=i, column=16, value="=" + "*".join(parts))
    chk.cell(row=i, column=17, value=f"=P{i}*B{i}")
chk.freeze_panes = "A2"
for col in range(1, 18):
    chk.column_dimensions[chr(64 + col)].width = 12

# ---------- reconciliation ----------
rec["A1"] = f"Actual vs predicted claims by bonus-malus band ({n:,} test-set policies)"
rec["A1"].font = Font(bold=True, size=14)
header(rec, 3, ["BM band", "Policies", "Exposure", "Actual claims", "Predicted claims",
                "Actual / predicted", "Actual freq", "Predicted freq"])
last_chk = n + 1
for i, lab in enumerate(BM_LAB, start=4):
    rec.cell(row=i, column=1, value=lab)
    rec.cell(row=i, column=2, value=f"=COUNTIF(Check!$O$2:$O${last_chk},\"=\"&A{i})")
    rec.cell(row=i, column=3, value=f"=SUMIFS(Check!$B$2:$B${last_chk},Check!$O$2:$O${last_chk},\"=\"&A{i})")
    rec.cell(row=i, column=4, value=f"=SUMIFS(Check!$A$2:$A${last_chk},Check!$O$2:$O${last_chk},\"=\"&A{i})")
    rec.cell(row=i, column=5, value=f"=SUMIFS(Check!$Q$2:$Q${last_chk},Check!$O$2:$O${last_chk},\"=\"&A{i})")
tot = 4 + len(BM_LAB)
rec.cell(row=tot, column=1, value="Total").font = Font(bold=True)
for col in "BCDE":
    rec[f"{col}{tot}"] = f"=SUM({col}4:{col}{tot - 1})"
    rec[f"{col}{tot}"].font = Font(bold=True)
for i in range(4, tot + 1):
    rec[f"F{i}"] = f"=IF(E{i}=0,\"\",D{i}/E{i})"
    rec[f"G{i}"] = f"=IF(C{i}=0,\"\",D{i}/C{i})"
    rec[f"H{i}"] = f"=IF(C{i}=0,\"\",E{i}/C{i})"
    rec[f"C{i}"].number_format = "#,##0.0"; rec[f"E{i}"].number_format = "#,##0.0"
    rec[f"F{i}"].number_format = "0.000"; rec[f"G{i}"].number_format = "0.0000"; rec[f"H{i}"].number_format = "0.0000"
rec[f"A{tot + 2}"] = ("The total ratio is the off-balance factor. If it is not near 1.00, rebalance the base rate "
                      "by multiplying the intercept by that ratio. Bands with few claims are noisy.")
for col, w in zip("ABCDEFGH", (14, 11, 12, 14, 16, 18, 13, 15)):
    rec.column_dimensions[col].width = w

import os
os.makedirs("excel", exist_ok=True)
wb.save("excel/auto_pricing_calculator.xlsx")
print("saved excel/auto_pricing_calculator.xlsx")

# ---------- python cross-check (should match the Reconciliation total after Excel recalculates) ----------
def band(x, up, lab):
    return pd.cut(x, [-np.inf] + up, labels=lab, right=True).astype(str)
p = pd.Series(rel("Freq", "Intercept", "base"), index=s.index)
cats = {"DrivAgeBand": band(s.DrivAge, AGE_UP, AGE_LAB), "VehAgeBand": band(s.VehAge, VA_UP, VA_LAB),
        "VehPowerC": s.VehPower.clip(4, 10).astype(int).astype(str), "VehGas": s.VehGas,
        "VehBrand": s.VehBrand, "Region": s.Region, "Area": s.Area,
        "BMBand": band(s.BonusMalus.clip(upper=150), BM_UP, BM_LAB)}
for var, ser in cats.items():
    if var in VARS["Freq"]:
        vals = ser.map(lambda l, var=var: D.get(("Freq", var, l), np.nan))
        if vals.isna().any():
            print(f"WARNING: {int(vals.isna().sum())} policies have {var} levels not found in the model: "
                  f"{sorted(set(ser[vals.isna()]))[:5]}")
        p = p * vals
if "LogDensity" in VARS["Freq"]:
    p = p * D[("Freq", "LogDensity", "per unit")] ** np.log(s.Density)
pred = (p * s.Exposure)
print(f"\nPython cross-check on {n:,} test policies: actual claims {s.ClaimNb.sum():,.0f}, "
      f"predicted {pred.sum():,.1f}, actual/predicted {s.ClaimNb.sum() / pred.sum():.3f}")
