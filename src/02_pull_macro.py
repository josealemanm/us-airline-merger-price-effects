"""Stage 2 - the two macro series the analysis needs.

CPI-U converts nominal fares to constant 2019 dollars. A fare comparison across
a fifteen-year window is meaningless without it: a 2005 ticket and a 2019 ticket
at the same dollar price are not the same price.

Gulf Coast jet fuel is a cost shifter. It moves for reasons that have nothing to
do with how much anyone wants to fly between two particular cities, which is
what makes it usable as an instrument once it is interacted with route distance:
a fuel price rise raises the cost of a long route more than a short one, and
that differential is the variation the demand estimation leans on.

Both come from FRED as plain CSV, no key required.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
SERIES = {
    "CPIAUCSL": "cpi",          # CPI for all urban consumers, all items, monthly, SA
    "WJFUELUSGULF": "jetfuel",  # US Gulf Coast kerosene-type jet fuel spot, weekly
}


def fetch(series: str) -> pd.DataFrame:
    for attempt in range(4):
        try:
            r = requests.get(FRED.format(series=series), timeout=90)
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna()
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry {series}: {exc}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def to_quarterly(df: pd.DataFrame, name: str) -> pd.DataFrame:
    q = df.set_index("date")["value"].resample("QE").mean().reset_index()
    q["year"] = q["date"].dt.year
    q["quarter"] = q["date"].dt.quarter
    return q[["year", "quarter", "value"]].rename(columns={"value": name})


def main() -> int:
    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    frames, meta = [], {}
    for series, name in SERIES.items():
        raw = fetch(series)
        meta[series] = {"name": name, "first": str(raw["date"].min().date()),
                        "last": str(raw["date"].max().date()), "n_obs": len(raw)}
        frames.append(to_quarterly(raw, name))
        print(f"  {series:<14} {name:<8} {len(raw):>6,} observations, "
              f"{raw['date'].min().date()} to {raw['date'].max().date()}", flush=True)

    macro = frames[0]
    for f in frames[1:]:
        macro = macro.merge(f, on=["year", "quarter"], how="outer")
    macro = macro.sort_values(["year", "quarter"])

    lo, hi = C.quarters()[0], C.quarters()[-1]
    macro = macro[(macro["year"] * 10 + macro["quarter"] >= lo[0] * 10 + lo[1])
                  & (macro["year"] * 10 + macro["quarter"] <= hi[0] * 10 + hi[1])]

    # Base period for real fares: 2019, the last full year in the window.
    base = macro.loc[macro["year"] == 2019, "cpi"].mean()
    macro["deflator"] = base / macro["cpi"]
    macro["qindex"] = [C.qindex(int(y), int(q))
                       for y, q in zip(macro["year"], macro["quarter"])]

    out = C.DATA_INTERIM / "macro.parquet"
    macro.to_parquet(out, index=False)
    (C.DATA_INTERIM / "macro_source.json").write_text(json.dumps({
        "source": "Federal Reserve Bank of St Louis, FRED",
        "series": meta,
        "cpi_base": "2019 annual average",
        "pulled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    print(f"\n  {len(macro)} quarters written to {out.name}")
    print(f"  deflator ranges {macro['deflator'].min():.3f} to {macro['deflator'].max():.3f} "
          f"(2019 = 1.000)")
    print(f"  jet fuel ranges ${macro['jetfuel'].min():.2f} to "
          f"${macro['jetfuel'].max():.2f} per gallon")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
