import argparse, re, unicodedata
import pandas as pd
from pathlib import Path

# mapowanie: nazwa pliku -> nazwa kolumny wynikowej
FILE_VAR = {
    "crime.csv": "crime",
    "unemployment.csv": "unemployment",
    "density.csv": "population_density",
    "salary.csv": "avg_salary",
    "safety_exp.csv": "public_safety_exp",
    "education_exp.csv": "education_exp",
    "migration.csv": "migration_balance",
    "tourism.csv": "tourism_usage",
    "inflation.csv": "inflation",
}

def strip_prefix_woj(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.strip()
    s = re.sub(r"^województwo |^wojewodztwo ", "", s, flags=re.I)
    return s

def coerce_num(x):
    if isinstance(x, str):
        x = x.replace("\xa0", " ").replace(" ", "").replace(",", ".")
    return pd.to_numeric(x, errors="coerce")

def load_bdl_wide(path: Path, varname: str) -> pd.DataFrame:
    # CSV z BDL „tablica wielowymiarowa”
    df = pd.read_csv(path, sep=None, engine="python")
    # usuń puste kolumny typu Unnamed
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
    # kolumna z nazwą jednostki (województwo)
    unit_col = "Nazwa" if "Nazwa" in df.columns else df.columns[0]
    # kolumny-lata mają wzorzec ;YYYY;
    year_cols = [c for c in df.columns if re.search(r";\d{4};", c)]
    if len(year_cols) < 2:
        raise ValueError(f"{path.name}: nie znaleziono kolumn lat (;YYYY;).")

    # z szerokiego na długi
    long = df.melt(id_vars=[unit_col], value_vars=year_cols,
                   var_name="__col__", value_name=varname)
    long["year"] = long["__col__"].str.extract(r";(\d{4});").astype(int)
    long = long.drop(columns="__col__")
    long = long.rename(columns={unit_col: "unit"})

    # czyszczenie
    long["unit"] = long["unit"].astype(str).map(strip_prefix_woj).str.strip().str.title()
    long[varname] = long[varname].map(coerce_num)

    # agregacja na wszelki wypadek
    long = long.groupby(["year", "unit"], as_index=False)[varname].mean()
    return long

def merge_all(in_dir: Path) -> pd.DataFrame:
    merged = None
    for fname, var in FILE_VAR.items():
        fpath = in_dir / fname
        if not fpath.exists():
            print(f"[WARN] Brak pliku: {fname} – pomijam.")
            continue
        print(f"[LOAD] {fname} -> {var}")
        df = load_bdl_wide(fpath, var)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["year", "unit"], how="outer")
    if merged is None:
        raise SystemExit("Nie wczytano żadnego pliku.")
    merged = merged.sort_values(["year", "unit"]).reset_index(drop=True)
    return merged

def main():
    ap = argparse.ArgumentParser(description="Preprocessing CSV z BDL (tablica wielowymiarowa) -> panel.csv")
    ap.add_argument("--in", dest="in_dir", default="../data/raw", help="folder z wejściowymi CSV")
    ap.add_argument("--out", dest="out_path", default="../data/processed/panel.csv", help="ścieżka wyjściowego CSV")
    ap.add_argument("--years", nargs=2, type=int, default=[2002, 2024], help="zakres lat [od do]")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel = merge_all(in_dir)

    y0, y1 = args.years
    panel = panel[(panel["year"] >= y0) & (panel["year"] <= y1)].copy()

    panel.to_csv(out_path, index=False)
    print(f"[OK] Zapisano: {out_path} ({len(panel)} wierszy, {panel.shape[1]} kolumn)")

    miss = panel.isna().mean().mul(100).round(2).to_frame("missing_%")
    miss_path = out_path.parent / "_missing_report.csv"
    miss.to_csv(miss_path)
    print(f"[OK] Raport braków: {miss_path}")

if __name__ == "__main__":
    main()
