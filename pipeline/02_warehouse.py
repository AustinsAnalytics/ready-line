"""Stage 2 - load the Parquet lake into a DuckDB analytical warehouse (star schema)
and answer the first descriptive questions in SQL.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import pandas as pd

from common import ENGINE_CHAPTERS, JASC_CHAPTERS, PROCESSED, WAREHOUSE, Timer, export


def main() -> None:
    if WAREHOUSE.exists():
        WAREHOUSE.unlink()
    con = duckdb.connect(str(WAREHOUSE))

    with Timer("warehouse build"):
        dim_jasc = pd.DataFrame(
            [{"chapter": k, "title": v, "is_engine": k in ENGINE_CHAPTERS} for k, v in JASC_CHAPTERS.items()]
        )
        con.register("dim_jasc_df", dim_jasc)
        con.execute("CREATE TABLE dim_jasc AS SELECT * FROM dim_jasc_df")

        con.execute(f"""
            CREATE TABLE fact_sdr AS
            SELECT * FROM read_parquet('{PROCESSED / "sdr"}/*/*.parquet', hive_partitioning = true)
        """)
        con.execute(f"""
            CREATE TABLE fact_engine_cycle AS
            SELECT * FROM read_parquet('{PROCESSED / "cmapss_features"}/*/*.parquet', hive_partitioning = true)
        """)
        con.execute("""
            CREATE TABLE dim_engine AS
            SELECT engine_id, subset, split, unit,
                   max(cycle)                       AS observed_cycles,
                   max(cycle) + coalesce(max(rul_at_end), 0) AS total_life,
                   bool_or(failed)                  AS run_to_failure,
                   max(rul_at_end)                  AS rul_at_end,
                   count(DISTINCT regime)           AS regimes_visited
            FROM fact_engine_cycle
            GROUP BY ALL
        """)
        con.execute("""
            CREATE VIEW v_sdr_monthly AS
            SELECT month, chapter, count(*) AS reports, sum(interruption::INT) AS interruptions
            FROM fact_sdr GROUP BY ALL
        """)

    tables = con.execute("""
        SELECT table_name, estimated_size AS rows, column_count AS cols
        FROM duckdb_tables() ORDER BY table_name
    """).df()
    print(tables)

    # ---- descriptive questions, answered in SQL --------------------------------
    by_year = con.execute("""
        SELECT year, count(*) AS reports,
               sum(is_engine::INT) AS engine_reports,
               sum(interruption::INT) AS interruptions
        FROM fact_sdr GROUP BY year ORDER BY year
    """).df()

    by_chapter = con.execute("""
        SELECT s.chapter, coalesce(j.title, 'Chapter ' || s.chapter) AS title,
               coalesce(j.is_engine, false) AS is_engine,
               count(*) AS reports,
               avg(s.interruption::INT) AS interruption_rate,
               avg(CASE WHEN s.stage_group NOT IN ('Inspection / maintenance', 'Other / unknown') THEN 1 ELSE 0 END) AS inflight_share
        FROM fact_sdr s LEFT JOIN dim_jasc j USING (chapter)
        GROUP BY ALL ORDER BY reports DESC
    """).df()

    makes = con.execute("""
        SELECT aircraft_make, count(*) AS reports FROM fact_sdr
        GROUP BY ALL ORDER BY reports DESC LIMIT 12
    """).df()

    precaution = con.execute("""
        SELECT precaution_a AS code, count(*) AS reports
        FROM fact_sdr WHERE precaution_a IS NOT NULL GROUP BY ALL ORDER BY reports DESC
    """).df()

    actions = con.execute("""
        SELECT is_engine, action, count(*) AS reports
        FROM fact_sdr GROUP BY ALL ORDER BY is_engine, reports DESC
    """).df()

    engines = con.execute("""
        SELECT subset, split, count(*) AS engines, avg(total_life) AS mean_life,
               min(total_life) AS min_life, max(total_life) AS max_life,
               avg(observed_cycles) AS mean_observed
        FROM dim_engine GROUP BY ALL ORDER BY subset, split
    """).df()

    lockheed = con.execute("""
        SELECT aircraft_model, count(*) AS reports FROM fact_sdr
        WHERE aircraft_make = 'LKHEED' GROUP BY ALL ORDER BY reports DESC LIMIT 6
    """).df()
    print(by_chapter.head(15))
    print(engines)
    print(lockheed)

    sql_example = """SELECT s.chapter, j.title,
       count(*)                      AS reports,
       avg(s.interruption::INT)      AS interruption_rate
FROM   fact_sdr s
JOIN   dim_jasc j USING (chapter)
WHERE  j.is_engine
GROUP  BY ALL
ORDER  BY interruption_rate DESC;"""

    export("warehouse", {
        "tables": tables.to_dict("records"),
        "by_year": by_year.to_dict("records"),
        "by_chapter": by_chapter.to_dict("records"),
        "makes": makes.to_dict("records"),
        "precaution": precaution.to_dict("records"),
        "actions": actions.to_dict("records"),
        "engines": engines.to_dict("records"),
        "lockheed": lockheed.to_dict("records"),
        "sql_example": sql_example,
        "db_megabytes": WAREHOUSE.stat().st_size / 1e6,
    })
    con.close()


if __name__ == "__main__":
    main()
