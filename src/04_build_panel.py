"""Stage 4 - assemble the analysis panel and define who was treated.

Three tables come out of this stage.

``panel_market``   one row per city pair per quarter: the passenger-weighted
                   real fare, total traffic, concentration, and how many
                   carriers were selling.

``panel_carrier``  one row per carrier per city pair per quarter, with the
                   shares the demand model needs, including the share of the
                   potential market that flew nobody - the outside option.

``treatment``      one row per city pair per merger, recording whether both
                   merging carriers were selling there before the deal was
                   announced, and with what shares.

The treatment definition is the load-bearing choice. A route counts as an
overlap route for a merger when *both* parties held at least one per cent of it,
averaged over the four quarters before the merger was announced. Measuring
before the announcement rather than before the closing matters: carriers move
capacity around once a deal is public, and a baseline measured after that would
be partly an outcome of the merger rather than a description of the market it
walked into.

Routes where exactly one party was selling are set aside rather than used as
controls. They are the obvious place for spillovers - the merged carrier
re-routing traffic through a hub it now owns - so they are held out and then
used as a placebo in stage 6.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def build_labels(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Readable route names: each city market takes its busiest airport code."""
    glob = (C.DATA_RAW / "xwalk_*.parquet").as_posix()
    return con.execute(f"""
        WITH x AS (
            SELECT city, airport, ANY_VALUE(state) AS state, SUM(pax) AS pax
            FROM read_parquet('{glob}') GROUP BY 1,2),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY city ORDER BY pax DESC) rn
            FROM x)
        SELECT city, airport, state FROM ranked WHERE rn = 1
    """).df()


def main() -> int:
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")

    glob = (C.DATA_RAW / "city_*.parquet").as_posix()
    macro = pd.read_parquet(C.DATA_INTERIM / "macro.parquet")
    con.register("macro", macro)

    # ----------------------------------------------------------- raw to real
    con.execute(f"""
        CREATE TABLE cc AS
        SELECT
            c.year, c.quarter,
            (c.year - 2000) * 4 + (c.quarter - 1) AS qindex,
            c.city1, c.city2, c.carrier,
            SUM(c.pax)                                   AS pax,
            SUM(c.fare_pax) * ANY_VALUE(m.deflator) / SUM(c.pax) AS fare,
            SUM(c.pax_nonstop) / SUM(c.pax)              AS nonstop_share,
            SUM(c.nsmiles_pax) / SUM(c.pax)              AS distance
        FROM read_parquet('{glob}') c
        JOIN macro m ON m.year = c.year AND m.quarter = c.quarter
        GROUP BY 1,2,3,4,5,6
    """)

    # --------------------------------------------------------- market totals
    con.execute(f"""
        CREATE TABLE mkt AS
        SELECT year, quarter, qindex, city1, city2,
               SUM(pax)                       AS pax,
               SUM(pax * fare) / SUM(pax)     AS fare,
               SUM(pax * nonstop_share) / SUM(pax) AS nonstop_share,
               SUM(pax * distance) / SUM(pax) AS distance,
               COUNT(*)                       AS n_carriers_raw
        FROM cc GROUP BY 1,2,3,4,5
        HAVING SUM(pax) >= {C.MIN_MARKET_PAX_PER_Q}
    """)

    # Endpoint traffic, the scale in the gravity proxy for potential market.
    con.execute("""
        CREATE TABLE endp AS
        SELECT qindex, city, SUM(pax) AS pax FROM (
            SELECT qindex, city1 AS city, pax FROM mkt
            UNION ALL
            SELECT qindex, city2 AS city, pax FROM mkt)
        GROUP BY 1,2
    """)
    con.execute("""
        CREATE TABLE qtot AS
        SELECT qindex, SUM(pax) AS pax FROM endp GROUP BY 1
    """)

    # ------------------------------------------------------- carrier shares
    # The gravity term P1 * P2 / Ptotal is the flow a standard gravity model
    # predicts between two cities of these sizes. It is used as the shape of the
    # potential market; the level is normalised below.
    con.execute(f"""
        CREATE TABLE car AS
        SELECT
            c.year, c.quarter, c.qindex, c.city1, c.city2, c.carrier,
            c.pax, c.fare, c.nonstop_share, c.distance,
            c.pax / m.pax                        AS share_inside,
            e1.pax * e2.pax / t.pax              AS gravity
        FROM cc c
        JOIN mkt m USING (qindex, city1, city2)
        JOIN endp e1 ON e1.qindex = c.qindex AND e1.city = c.city1
        JOIN endp e2 ON e2.qindex = c.qindex AND e2.city = c.city2
        JOIN qtot t ON t.qindex = c.qindex
        WHERE c.pax / m.pax >= {C.MIN_CARRIER_SHARE}
    """)

    # Re-normalise after dropping sub-1% carriers, so shares inside a market
    # still sum to one and the HHI is computed on the carriers that are actually
    # competing rather than on a tail of one-ticket entries.
    con.execute("""
        CREATE TABLE carrier_panel AS
        WITH tot AS (
            SELECT qindex, city1, city2, SUM(pax) AS pax_kept, COUNT(*) AS n_carriers
            FROM car GROUP BY 1,2,3)
        SELECT
            c.year, c.quarter, c.qindex, c.city1, c.city2, c.carrier,
            c.pax, c.fare, c.nonstop_share, c.distance,
            c.pax / t.pax_kept                       AS share,
            t.pax_kept                               AS market_pax,
            t.n_carriers,
            c.gravity
        FROM car c JOIN tot t USING (qindex, city1, city2)
        WHERE t.n_carriers >= 1
    """)

    # ------------------------------------------------------- market panel
    con.execute("""
        CREATE TABLE market_panel AS
        SELECT
            year, quarter, qindex, city1, city2,
            SUM(pax)                                  AS pax,
            SUM(pax * fare) / SUM(pax)                AS fare,
            ln(SUM(pax * fare) / SUM(pax))            AS log_fare,
            SUM(pax * nonstop_share) / SUM(pax)       AS nonstop_share,
            SUM(pax * distance) / SUM(pax)            AS distance,
            COUNT(*)                                  AS n_carriers,
            10000 * SUM(share * share)                AS hhi,
            MAX(share)                                AS top_share,
            ANY_VALUE(gravity)                        AS gravity,
            ANY_VALUE(market_pax)                     AS market_pax
        FROM carrier_panel GROUP BY 1,2,3,4,5
    """)

    mp = con.execute("SELECT * FROM market_panel").df()
    cp = con.execute("SELECT * FROM carrier_panel").df()
    mp["market_id"] = mp["city1"].astype(str) + "_" + mp["city2"].astype(str)
    cp["market_id"] = cp["city1"].astype(str) + "_" + cp["city2"].astype(str)
    cp["group"] = [C.carrier_group(c) for c in cp["carrier"]]

    # ------------------------------------------------- potential market size
    # The gravity term fixes the shape of the potential market across routes.
    # Its level is a normalisation and there is no observation that pins it,
    # because nobody records the passengers who did not fly. It is set so that
    # the median route sells to the share of its potential market given by
    # OUTSIDE_SHARE_TARGET. This is a stated convention, not an estimate, and
    # stage 7 reports how the demand parameters move when it is halved and
    # doubled. The simulated percentage fare changes are close to invariant to
    # it; the dollar figures are not.
    # The potential market is held fixed over time for a given route. This
    # matters more than it looks. A gravity term rebuilt every quarter from that
    # quarter's traffic moves up and down with the business cycle in step with
    # the numerator, so the share of people choosing not to fly barely changes -
    # and the outside option, which is the whole point of having one, stops
    # responding to anything. Fixing it at the route's own long-run average lets
    # a recession or a fuel spike show up as passengers leaving the market,
    # which is what the demand estimation in stage 7 needs to see.
    route_gravity = mp.groupby("market_id")["gravity"].mean()
    for df in (mp, cp):
        df["gravity_route"] = df["market_id"].map(route_gravity)
    ratio = (mp["market_pax"] / mp["gravity_route"]).median()
    kappa = float(ratio / C.INSIDE_SHARE_TARGET)
    for df in (mp, cp):
        df["market_size"] = df["gravity_route"] * kappa
    mp["inside_share"] = (mp["market_pax"] / mp["market_size"]).clip(upper=0.95)
    cp["inside_share"] = (cp["market_pax"] / cp["market_size"]).clip(upper=0.95)
    cp["share_total"] = cp["share"] * cp["inside_share"]
    mp["outside_share"] = 1.0 - mp["inside_share"]
    cp["outside_share"] = 1.0 - cp["inside_share"]
    n_clipped = int((mp["market_pax"] / mp["market_size"] > 0.95).sum())

    # --------------------------------------------------------- route labels
    lab = build_labels(con).set_index("city")
    code = lab["airport"].to_dict()
    for df in (mp, cp):
        df["route"] = [f"{code.get(a, a)}-{code.get(b, b)}"
                       for a, b in zip(df["city1"], df["city2"])]

    # ------------------------------------------------------------ treatment
    rows = []
    for m in C.MERGERS:
        lo = m.announce_q - C.BASELINE_QUARTERS
        hi = m.announce_q - 1
        base = cp[(cp["qindex"] >= lo) & (cp["qindex"] <= hi)]
        nq = base.groupby("market_id")["qindex"].nunique()
        # Average share over the baseline window, counting quarters the carrier
        # was absent as zero rather than skipping them.
        piv = (base[base["carrier"].isin(m.carriers)]
               .groupby(["market_id", "carrier"])["share"].sum()
               .unstack(fill_value=0.0))
        for c in m.carriers:
            if c not in piv.columns:
                piv[c] = 0.0
        piv = piv.div(nq.reindex(piv.index).clip(lower=1), axis=0)
        piv = piv.reindex(nq.index, fill_value=0.0)
        a, b = m.carriers
        present_a = piv[a] >= C.MIN_CARRIER_SHARE
        present_b = piv[b] >= C.MIN_CARRIER_SHARE
        n_parties = present_a.astype(int) + present_b.astype(int)
        # Markets must be observed through the baseline to be classified at all.
        ok = nq.reindex(piv.index).fillna(0) >= C.BASELINE_QUARTERS
        for mid in piv.index[ok]:
            rows.append({
                "merger": m.key, "market_id": mid,
                "share_a": float(piv.loc[mid, a]), "share_b": float(piv.loc[mid, b]),
                "n_parties": int(n_parties.loc[mid]),
                "overlap": bool(n_parties.loc[mid] == 2),
                "one_party": bool(n_parties.loc[mid] == 1),
                "neither": bool(n_parties.loc[mid] == 0),
                "close_q": m.close_q, "announce_q": m.announce_q,
            })
    tr = pd.DataFrame(rows)

    # A control route for one merger must not be an overlap route for any other
    # merger whose closing falls inside this one's event window. Otherwise the
    # "control" group is absorbing a different merger's effect.
    overlap_by_merger = {k: set(g.loc[g["overlap"], "market_id"])
                         for k, g in tr.groupby("merger")}
    def clean_control(row) -> bool:
        if not row["neither"]:
            return False
        me = C.MERGERS_BY_KEY[row["merger"]]
        for other in C.MERGERS:
            if other.key == me.key:
                continue
            if abs(other.close_q - me.close_q) <= C.EVENT_MAX - C.EVENT_MIN:
                if row["market_id"] in overlap_by_merger.get(other.key, set()):
                    return False
        return True
    tr["clean_control"] = tr.apply(clean_control, axis=1)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    import json
    (C.DATA_PROCESSED / "market_size_normalisation.json").write_text(json.dumps({
        "form": "potential market = kappa * P_endpoint1 * P_endpoint2 / P_total",
        "kappa": kappa,
        "inside_share_target_at_median": C.INSIDE_SHARE_TARGET,
        "median_inside_share_achieved": float(mp["inside_share"].median()),
        "market_quarters_clipped_at_0.95": n_clipped,
    }, indent=2))
    mp.to_parquet(C.DATA_PROCESSED / "panel_market.parquet", index=False)
    cp.to_parquet(C.DATA_PROCESSED / "panel_carrier.parquet", index=False)
    tr.to_parquet(C.DATA_PROCESSED / "treatment.parquet", index=False)

    # ------------------------------------------------------------- report
    print(f"  market panel   {len(mp):>10,} rows   "
          f"{mp['market_id'].nunique():>7,} routes   "
          f"{mp['qindex'].nunique()} quarters")
    print(f"  carrier panel  {len(cp):>10,} rows   "
          f"{cp['carrier'].nunique()} carriers")
    print(f"  real fare      mean ${mp['fare'].mean():.2f}  "
          f"median ${mp['fare'].median():.2f}  (2019 dollars)")
    print(f"  HHI            mean {mp['hhi'].mean():,.0f}  "
          f"median {mp['hhi'].median():,.0f}")
    print(f"  carriers/route mean {mp['n_carriers'].mean():.2f}")
    print(f"  inside share   mean {mp['inside_share'].mean():.3f}  "
          f"median {mp['inside_share'].median():.3f}   "
          f"(kappa = {kappa:.1f}, {n_clipped:,} market-quarters clipped at 0.95)")
    print()
    print("  treatment groups by merger:")
    print("  {:<8} {:>9} {:>11} {:>11} {:>14}".format(
        "merger", "overlap", "one party", "neither", "clean control"))
    for m in C.MERGERS:
        g = tr[tr["merger"] == m.key]
        print("  {:<8} {:>9,} {:>11,} {:>11,} {:>14,}".format(
            m.key, int(g["overlap"].sum()), int(g["one_party"].sum()),
            int(g["neither"].sum()), int(g["clean_control"].sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
