"""Compare two analysed reports (two years of one company, or two companies)."""
import pandas as pd

from .schemas import METRIC_KEYS


def canon(unit):
    """Return (base_unit, factor) to convert a value into a base unit, or (None, None) if unknown."""
    u = (unit or "").lower().replace("₂", "2").replace("³", "3").replace(" ", "").replace(".", "")
    if not u:
        return None, None
    if "co2" in u:                                   # tonnes of CO2 equivalent, tCO2e, ...
        return ("tCO2e", 0.001) if "kg" in u else ("tCO2e", 1.0)
    if u in ("%", "percent", "percentage"):
        return "%", 1.0
    energy = {"gj": 1.0, "gigajoule": 1.0, "gigajoules": 1.0, "tj": 1000.0, "terajoule": 1000.0,
              "terajoules": 1000.0, "mwh": 3.6, "kwh": 0.0036}
    if u in energy:
        return "GJ", energy[u]
    water = {"kl": 1.0, "kilolitre": 1.0, "kilolitres": 1.0, "kiloliter": 1.0, "kiloliters": 1.0,
             "m3": 1.0, "cubicmetre": 1.0, "cubicmetres": 1.0, "cubicmeter": 1.0, "cubicmeters": 1.0,
             "megalitre": 1000.0, "megalitres": 1000.0, "megaliter": 1000.0, "megaliters": 1000.0,
             "l": 0.001, "litre": 0.001, "litres": 0.001, "liter": 0.001, "liters": 0.001}
    if u in water:
        return "kL", water[u]
    if u.startswith("kg") or u.startswith("kilogram"):
        return "tonnes", 0.001
    if "tonne" in u or u in ("t", "ton", "tons"):
        return "tonnes", 1.0
    return None, None


def build_comparison(res_a: dict, res_b: dict, label_a: str, label_b: str) -> pd.DataFrame:
    rows = []
    for key in METRIC_KEYS:
        ma, mb = res_a["metrics"][key], res_b["metrics"][key]
        a, b = ma["value"], mb["value"]
        unit = mb["unit"] or ma["unit"]
        change = pp = None
        note = ""
        if a is None or b is None:
            note = "not disclosed in one report"
        else:
            ba, fa = canon(ma["unit"])
            bb, fb = canon(mb["unit"])
            if ba is None or bb is None:
                same = (ma["unit"] or "").lower() == (mb["unit"] or "").lower()
                if same:
                    ba = bb = unit
                    fa = fb = 1.0
            if ba is None or ba != bb:
                note = f"units differ ({ma['unit']} vs {mb['unit']}), not compared"
            else:
                if fa != 1.0 or fb != 1.0:
                    a, b = a * fa, b * fb
                    note = f"converted to {ba}"
                a, b, unit = round(a, 4), round(b, 4), ba
                if ba == "%":
                    pp = round(b - a, 2)          # change in percentage points, not % of a %
                    note = "change in percentage points"
                elif a != 0:
                    change = round((b - a) / a * 100, 2)
        rows.append({"metric": key, "unit": unit, label_a: a, label_b: b,
                     "change_%": change, "change_pp": pp, "note": note})
    for name, k, u in [("scope1_plus_scope2", "scope12_total", "tCO2e"),
                       ("emission_intensity", "intensity_tco2e_per_crore", "tCO2e per ₹ crore")]:
        a, b = res_a["analysis"][k], res_b["analysis"][k]
        change = round((b - a) / a * 100, 2) if a and b is not None else None
        rows.append({"metric": name, "unit": u, label_a: a, label_b: b, "change_%": change,
                     "change_pp": None,
                     "note": "" if change is not None else "needs both values (enter revenue for intensity)"})
    return pd.DataFrame(rows)
