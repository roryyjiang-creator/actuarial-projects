"""Build excel/reserve_discounting.xlsx: the Project 5 discounting model as live Excel formulas.

Inputs (blue cells) drive a Vasicek discount-factor table and two scenario tables. A Check sheet compares the
workbook's formulas with the same calculation done in Python. Run from the project5_interest_rates folder.
"""
import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.workbook.defined_name import DefinedName

cf = pd.read_csv("reports/outstanding_cashflows.csv")
prm = pd.read_csv("reports/vasicek_params.csv").set_index("parameter")["estimate"]
A, B, S = float(prm["speed a"]), float(prm["long-run mean b"]), float(prm["volatility sigma"])
R0 = float(pd.read_csv("data/short_rate.csv").short_rate.iloc[-1])
pay = cf["expected payment"].to_numpy()
years = cf["year from now"].to_numpy()
OFFSET, FLAT = 0.5, 0.04

def pv_py(rate, shift=0.0):
    t = years - OFFSET + shift
    Bt = (1 - np.exp(-A * t)) / A
    lnA = (Bt - t) * (A * A * B - S * S / 2) / (A * A) - S * S * Bt * Bt / (4 * A)
    return float((pay * np.exp(lnA - Bt * rate)).sum())

wb = Workbook()
blue = Font(color="0000FF")
inp = PatternFill("solid", fgColor="FFF2CC")
bold = Font(bold=True)
head = PatternFill("solid", fgColor="D9E1F2")

def header(ws, row, labels):
    for j, lab in enumerate(labels, 1):
        c = ws.cell(row, j, lab); c.font = bold; c.fill = head; c.alignment = Alignment(wrap_text=True)

# ---------------- Inputs ----------------
ws = wb.active; ws.title = "Inputs"
ws["A1"] = "Reserve discounting model: inputs"; ws["A1"].font = Font(bold=True, size=13)
ws["A2"] = "Blue cells on yellow are inputs. Everything else is formulas."
items = [("Vasicek speed a", A, "VasA", "0.0000"), ("Vasicek long-run mean b", B, "VasB", "0.0000"),
         ("Vasicek volatility sigma", S, "VasSig", "0.0000"), ("Current short rate", R0, "ShortRate", "0.00%"),
         ("Flat rate for comparison", FLAT, "FlatRate", "0.00%"),
         ("Payment time within the year (0.5 = mid-year)", OFFSET, "PayOffset", "0.00"),
         ("Shift all payments later by (years)", 0.0, "TimeShift", "0.00")]
for i, (lab, val, name, fmt) in enumerate(items, 3):
    ws.cell(i, 1, lab); c = ws.cell(i, 2, val); c.font = blue; c.fill = inp; c.number_format = fmt
    wb.defined_names[name] = DefinedName(name, attr_text=f"Inputs!$B${i}")
header(ws, 12, ["Year from now", "Expected payment"])
for k, (y, p) in enumerate(zip(years, pay)):
    r = 13 + k
    ws.cell(r, 1, int(y)); c = ws.cell(r, 2, float(p)); c.font = blue; c.fill = inp; c.number_format = "#,##0"
wb.defined_names["PayYear"] = DefinedName("PayYear", attr_text="Inputs!$A$13:$A$21")
wb.defined_names["ClaimPay"] = DefinedName("ClaimPay", attr_text="Inputs!$B$13:$B$21")
ws.column_dimensions["A"].width = 46; ws.column_dimensions["B"].width = 18

# ---------------- Model ----------------
m = wb.create_sheet("Model")
header(m, 1, ["Year", "Payment", "Time t (yrs)", "B(t)", "ln A(t)", "Vasicek discount factor", "PV (Vasicek)",
              "Flat discount factor", "PV (flat)"])
for k in range(9):
    r = 2 + k; i = 13 + k
    m[f"A{r}"] = f"=Inputs!A{i}"; m[f"B{r}"] = f"=Inputs!B{i}"
    m[f"C{r}"] = f"=A{r}-PayOffset+TimeShift"
    m[f"D{r}"] = f"=(1-EXP(-VasA*C{r}))/VasA"
    m[f"E{r}"] = f"=(D{r}-C{r})*(VasA^2*VasB-VasSig^2/2)/VasA^2-VasSig^2*D{r}^2/(4*VasA)"
    m[f"F{r}"] = f"=EXP(E{r}-D{r}*ShortRate)"
    m[f"G{r}"] = f"=B{r}*F{r}"
    m[f"H{r}"] = f"=EXP(-FlatRate*C{r})"
    m[f"I{r}"] = f"=B{r}*H{r}"
    for col, fmt in (("B", "#,##0"), ("C", "0.00"), ("D", "0.0000"), ("E", "0.0000"), ("F", "0.0000"),
                     ("G", "#,##0"), ("H", "0.0000"), ("I", "#,##0")):
        m[f"{col}{r}"].number_format = fmt
m["A11"] = "Total"; m["A11"].font = bold
m["B11"] = "=SUM(B2:B10)"; m["G11"] = "=SUM(G2:G10)"; m["I11"] = "=SUM(I2:I10)"
for c in ("B11", "G11", "I11"):
    m[c].number_format = "#,##0"; m[c].font = bold
summ = [("Undiscounted reserve", "=B11", "#,##0"), ("PV, Vasicek model", "=G11", "#,##0"),
        ("Discount, Vasicek", "=1-G11/B11", "0.0%"), ("PV, flat rate", "=I11", "#,##0"),
        ("Discount, flat rate", "=1-I11/B11", "0.0%"),
        ("Average time to payment, Vasicek PV-weighted (yrs)", "=SUMPRODUCT(C2:C10,G2:G10)/G11", "0.00")]
for k, (lab, f, fmt) in enumerate(summ):
    m.cell(14 + k, 1, lab); c = m.cell(14 + k, 2, f); c.number_format = fmt
m.column_dimensions["A"].width = 50
for col in "BCDEFGHI":
    m.column_dimensions[col].width = 16

# ---------------- Scenarios ----------------
def pv_formula(rate_ref, shift_ref):
    T = f"(PayYear-PayOffset+{shift_ref})"
    Bf = f"((1-EXP(-VasA*{T}))/VasA)"
    return (f"=SUMPRODUCT(ClaimPay,EXP(({Bf}-{T})*(VasA^2*VasB-VasSig^2/2)/VasA^2"
            f"-VasSig^2*{Bf}^2/(4*VasA)-{Bf}*{rate_ref}))")

sc = wb.create_sheet("Scenarios")
sc["A1"] = "A. Change in the starting short rate (Vasicek model)"; sc["A1"].font = bold
header(sc, 2, ["Change in short rate", "Short rate used", "PV", "vs base"])
deltas = [-0.02, -0.01, 0.0, 0.01, 0.02]
for k, d in enumerate(deltas):
    r = 3 + k
    c = sc.cell(r, 1, d); c.font = blue; c.fill = inp; c.number_format = "+0.0%;-0.0%;0.0%"
    sc[f"B{r}"] = f"=ShortRate+A{r}"; sc[f"B{r}"].number_format = "0.00%"
    sc[f"C{r}"] = pv_formula(f"B{r}", "TimeShift"); sc[f"C{r}"].number_format = "#,##0"
    sc[f"D{r}"] = f"=C{r}/$C$5-1"; sc[f"D{r}"].number_format = "+0.0%;-0.0%;0.0%"
sc["A10"] = "B. Payments earlier or later (Vasicek model, current short rate)"; sc["A10"].font = bold
header(sc, 11, ["Shift (years)", "", "PV", "vs base"])
shifts = [-0.5, 0.0, 0.5, 1.0]
for k, d in enumerate(shifts):
    r = 12 + k
    c = sc.cell(r, 1, d); c.font = blue; c.fill = inp; c.number_format = "+0.0;-0.0;0.0"
    sc[f"C{r}"] = pv_formula("ShortRate", f"A{r}"); sc[f"C{r}"].number_format = "#,##0"
    sc[f"D{r}"] = f"=C{r}/$C$13-1"; sc[f"D{r}"].number_format = "+0.0%;-0.0%;0.0%"
sc["A18"] = "Rows marked 0 are the base case. Table A uses the TimeShift input, table B ignores it."
sc.column_dimensions["A"].width = 24
for col in "BCD":
    sc.column_dimensions[col].width = 18

# ---------------- Check ----------------
ck = wb.create_sheet("Check")
header(ck, 1, ["Item", "Excel formula", "Python", "Difference", "OK?"])
rows = [("PV base (Vasicek)", "=Model!G11", pv_py(R0))]
rows += [(f"PV, short rate {d:+.0%}", f"=Scenarios!C{3 + k}", pv_py(R0 + d)) for k, d in enumerate(deltas)]
rows += [(f"PV, payments shifted {s:+.1f} yrs", f"=Scenarios!C{12 + k}", pv_py(R0, s)) for k, s in enumerate(shifts)]
rows += [("PV, flat rate", "=Model!I11", float((pay * np.exp(-FLAT * (years - OFFSET))).sum()))]
for k, (lab, f, val) in enumerate(rows):
    r = 2 + k
    ck[f"A{r}"] = lab; ck[f"B{r}"] = f; ck[f"C{r}"] = val
    ck[f"D{r}"] = f"=B{r}-C{r}"; ck[f"E{r}"] = f'=IF(ABS(D{r})<0.5,"OK","CHECK")'
    for col in "BC":
        ck[f"{col}{r}"].number_format = "#,##0.00"
    ck[f"D{r}"].number_format = "0.0000"
n = 2 + len(rows)
ck[f"A{n + 1}"] = "All checks pass?"; ck[f"A{n + 1}"].font = bold
ck[f"B{n + 1}"] = f'=IF(COUNTIF(E2:E{n - 1},"OK")={len(rows)},"YES","NO")'
ck.column_dimensions["A"].width = 34
for col in "BCDE":
    ck.column_dimensions[col].width = 18

os.makedirs("excel", exist_ok=True)
wb.save("excel/reserve_discounting.xlsx")
print(f"saved excel/reserve_discounting.xlsx; Python base PV {pv_py(R0):,.0f}")
