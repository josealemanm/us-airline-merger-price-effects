"""Stage 1 - pull the DOT Origin & Destination Survey (DB1B) Market file.

One zip per quarter from the Bureau of Transportation Statistics. Each unpacks
to roughly three million ticket records, which is too much to keep sixty times
over, so every quarter is aggregated to its analysis grain on arrival and the
raw file is deleted. What survives is two parquet files per quarter:

    city-pair    x ticketing carrier   the grain the analysis runs on
    airport-pair x ticketing carrier   kept for the market-definition robustness check

The manifest records the URL, the byte count, the SHA-256 of the zip and the
row counts in and out, so that a later run can prove it read the same bytes.

Usage:  python src/01_pull_db1b.py [--workers N] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

BASE = "https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_{year}_{quarter}.zip"
MANIFEST = C.DATA_RAW / "manifest.json"
SCRATCH = C.DATA_RAW / "_scratch"

# The columns we keep. DB1B Market carries forty-one; the rest are duplicates of
# these under other encodings, or geography the analysis does not use.
KEEP = """
    Year, Quarter, OriginCityMarketID, DestCityMarketID, Origin, Dest,
    TkCarrier, Passengers, MktFare, MktCoupons, MktDistance, NonStopMiles,
    BulkFare, OriginCountry, DestCountry, OriginState, DestState
"""

# Filters applied before aggregation. Each is a standard screen in the airline
# pricing literature; config.py holds the cut-offs and the reasoning.
WHERE = f"""
    OriginCountry = 'US' AND DestCountry = 'US'
    AND BulkFare = 0
    AND Passengers >= 1
    AND MktFare BETWEEN {C.MIN_FARE_NOMINAL} AND {C.MAX_FARE_NOMINAL}
    AND MktCoupons <= {C.MAX_COUPONS}
    AND TkCarrier IS NOT NULL
    AND OriginCityMarketID <> DestCityMarketID
"""

# Fares in DB1B are per passenger on that market record, so every average below
# is passenger-weighted. Distance is weighted the same way.
AGG = """
    SUM(Passengers)                                          AS pax,
    SUM(Passengers * MktFare)                                AS fare_pax,
    SUM(CASE WHEN MktCoupons = 1 THEN Passengers ELSE 0 END) AS pax_nonstop,
    SUM(Passengers * NonStopMiles)                           AS nsmiles_pax,
    SUM(Passengers * MktDistance)                            AS miles_pax,
    COUNT(*)                                                 AS n_records
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, tries: int = 5) -> int:
    """Stream a zip to disk, retrying with exponential backoff."""
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 600)) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if dest.stat().st_size < 1_000_000:
                raise OSError(f"suspiciously small download: {dest.stat().st_size} bytes")
            return dest.stat().st_size
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            dest.unlink(missing_ok=True)
            if attempt == tries:
                raise
            wait = min(60, 2 ** attempt)
            print(f"  retry {attempt}/{tries} after {type(exc).__name__}: {exc}; "
                  f"sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _aggregate(csv_path: Path, year: int, quarter: int) -> dict:
    """Collapse one quarter of ticket records to the two analysis grains."""
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA disable_progress_bar")
    # The CSV is read exactly once. Reading it through a view instead would
    # re-parse 700 MB for every query below, which costs four extra minutes a
    # quarter and four hours over the window.
    src = (
        f"read_csv('{csv_path.as_posix()}', header=true, "
        "sample_size=-1, ignore_errors=true, null_padding=true)"
    )
    con.execute(f"CREATE TABLE raw AS SELECT {KEEP} FROM {src}")
    rows_in = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    con.execute(f"CREATE TABLE kept AS SELECT * FROM raw WHERE {WHERE}")
    con.execute("DROP TABLE raw")
    rows_kept = con.execute("SELECT COUNT(*) FROM kept").fetchone()[0]

    city = C.DATA_RAW / f"city_{year}Q{quarter}.parquet"
    con.execute(f"""
        COPY (
            SELECT
                Year AS year, Quarter AS quarter,
                LEAST(OriginCityMarketID, DestCityMarketID)    AS city1,
                GREATEST(OriginCityMarketID, DestCityMarketID) AS city2,
                TkCarrier AS carrier,
                {AGG}
            FROM kept
            GROUP BY 1,2,3,4,5
        ) TO '{city.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # A city market can hold several airports (New York holds JFK, LGA and EWR).
    # The airport-pair file lets the market-definition robustness check re-run
    # on the narrower definition without pulling the raw data again.
    airport = C.DATA_RAW / f"airport_{year}Q{quarter}.parquet"
    con.execute(f"""
        COPY (
            SELECT
                Year AS year, Quarter AS quarter,
                LEAST(Origin, Dest)    AS ap1,
                GREATEST(Origin, Dest) AS ap2,
                TkCarrier AS carrier,
                {AGG}
            FROM kept
            GROUP BY 1,2,3,4,5
        ) TO '{airport.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # City market to busiest airport, used only to put readable labels on routes.
    xwalk = C.DATA_RAW / f"xwalk_{year}Q{quarter}.parquet"
    con.execute(f"""
        COPY (
            SELECT OriginCityMarketID AS city, Origin AS airport,
                   ANY_VALUE(OriginState) AS state, SUM(Passengers) AS pax
            FROM kept GROUP BY 1,2
        ) TO '{xwalk.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    rows_city = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{city.as_posix()}')").fetchone()[0]
    rows_ap = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{airport.as_posix()}')").fetchone()[0]
    con.close()
    return {
        "rows_in": rows_in, "rows_kept": rows_kept,
        "rows_city": rows_city, "rows_airport": rows_ap,
    }


def pull_quarter(year: int, quarter: int, force: bool) -> dict:
    out = C.DATA_RAW / f"city_{year}Q{quarter}.parquet"
    if out.exists() and not force:
        return {"year": year, "quarter": quarter, "status": "cached"}

    url = BASE.format(year=year, quarter=quarter)
    work = SCRATCH / f"{year}Q{quarter}"
    work.mkdir(parents=True, exist_ok=True)
    zpath = work / "db1b.zip"
    t0 = time.time()
    try:
        nbytes = _download(url, zpath)
        digest = _sha256(zpath)
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            z.extract(name, work)
        stats = _aggregate(work / name, year, quarter)
        rec = {
            "year": year, "quarter": quarter, "status": "ok", "url": url,
            "bytes": nbytes, "sha256": digest,
            "seconds": round(time.time() - t0, 1), **stats,
        }
        print(f"  {year}Q{quarter}  {nbytes/1e6:7.1f} MB  "
              f"{stats['rows_in']:>9,} rows -> {stats['rows_kept']:>9,} kept -> "
              f"{stats['rows_city']:>7,} city rows  ({rec['seconds']}s)", flush=True)
        return rec
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    qs = C.quarters()
    print(f"DB1B Market: {len(qs)} quarters, {qs[0][0]}Q{qs[0][1]} to "
          f"{qs[-1][0]}Q{qs[-1][1]}, {args.workers} workers", flush=True)

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(pull_quarter, y, q, args.force): (y, q) for y, q in qs}
        for fut in as_completed(futs):
            y, q = futs[fut]
            try:
                records.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {y}Q{q}: {type(exc).__name__}: {exc}", flush=True)
                failures.append({"year": y, "quarter": q, "error": repr(exc)})

    shutil.rmtree(SCRATCH, ignore_errors=True)

    # A quarter that was already on disk is reported as "cached" and carries no
    # byte count, digest or row counts. Writing those straight to the manifest
    # would erase what an earlier run recorded, so the previous entry is carried
    # forward instead. Without this, re-pulling a single quarter silently
    # destroys the provenance of the other fifty-nine.
    previous = {}
    if MANIFEST.exists():
        try:
            for rec in json.loads(MANIFEST.read_text()).get("quarters", []):
                if rec.get("status") == "ok":
                    previous[(rec["year"], rec["quarter"])] = rec
        except (json.JSONDecodeError, KeyError):
            pass
    records = [previous.get((r["year"], r["quarter"]), r)
               if r.get("status") == "cached" else r
               for r in records]
    records.sort(key=lambda r: (r["year"], r["quarter"]))
    MANIFEST.write_text(json.dumps({
        "source": "US DOT Bureau of Transportation Statistics, Airline Origin "
                  "and Destination Survey (DB1B), Market file",
        "url_pattern": BASE,
        "pulled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "filters": " ".join(WHERE.split()),
        "quarters": records,
        "failures": failures,
    }, indent=2))

    ok = [r for r in records if r.get("status") == "ok"]
    print(f"\ndone: {len(ok)} pulled, {len(records)-len(ok)} cached, "
          f"{len(failures)} failed")
    if ok:
        print(f"total download: {sum(r['bytes'] for r in ok)/1e9:.2f} GB")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
