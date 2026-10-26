"""Stage 1 - distributed ETL with PySpark.

* FAA Service Difficulty Reports (10 yearly CSVs, ~624k records, multi-line free text)
  -> cleaned, typed, partitioned Parquet + monthly aggregates.
* NASA C-MAPSS (4 subsets x train/test, space-delimited text)
  -> one unified engine-cycle table with RUL labels, operating-regime clusters
     (Spark ML k-means), regime-normalized sensors and rolling-window features.

Runs in Spark local mode here; the same code runs unchanged on a cluster
(spark-submit --master yarn ...), which is the point.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


from common import (CMAPSS_COLS, ENGINE_CHAPTERS, INTERRUPTION_CODES, PROCESSED, RAW,
                    SENSORS, SETTINGS, STAGE_GROUPS, SUBSETS, Timer, export)

INFORMATIVE = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]


def spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("ready-line-etl")
        .master("local[*]")
        .config("spark.driver.memory", "3g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


# --------------------------------------------------------------------------- SDR
def etl_sdr(spark: SparkSession) -> dict:
    raw = (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .option("mode", "PERMISSIVE")
        .csv(str(RAW / "sdr" / "SDR-*.csv"))
        .withColumn("source_file", F.regexp_extract(F.input_file_name(), r"SDR-(\d{4})", 1))
    )
    n_raw = raw.count()

    stage_map = F.create_map(*[F.lit(x) for kv in STAGE_GROUPS.items() for x in kv])
    interrupt = F.array(*[F.lit(c) for c in sorted(INTERRUPTION_CODES)])
    num = lambda c: F.regexp_replace(F.col(c), ",", "").cast("double")  # noqa: E731

    sdr = (
        raw.select(
            F.col("OperatorControlNumber").alias("report_id"),
            F.to_date("DifficultyDate", "MM/dd/yyyy").alias("difficulty_date"),
            F.col("source_file").cast("int").alias("file_year"),
            F.col("SubmitterTypeCode").alias("submitter_type"),
            F.lpad(F.trim(F.col("JASCCode")), 4, "0").alias("jasc"),
            F.upper(F.trim(F.col("NatureOfConditionA"))).alias("nature_a"),
            F.upper(F.trim(F.col("PrecautionaryProcedureA"))).alias("precaution_a"),
            F.upper(F.trim(F.col("PrecautionaryProcedureB"))).alias("precaution_b"),
            F.upper(F.trim(F.col("StageOfOperationCode"))).alias("stage"),
            F.upper(F.trim(F.col("HowDiscoveredCode"))).alias("how_discovered"),
            F.upper(F.trim(F.col("AircraftMake"))).alias("aircraft_make"),
            F.upper(F.trim(F.col("AircraftModel"))).alias("aircraft_model"),
            num("AircraftTotalTime").alias("aircraft_hours"),
            F.upper(F.trim(F.col("EngineMake"))).alias("engine_make"),
            F.upper(F.trim(F.col("EngineModel"))).alias("engine_model"),
            F.upper(F.trim(F.col("PartName"))).alias("part_name"),
            F.upper(F.trim(F.col("PartCondition"))).alias("part_condition"),
            num("PartTotalTime").alias("part_hours"),
            num("PartTimeSince").alias("part_time_since"),
            F.col("PartSinceCode").alias("part_since_code"),
            F.regexp_replace(F.col("Discrepancy"), r"\s+", " ").alias("text"),
        )
        .withColumn("year", F.year("difficulty_date"))
        .withColumn("month", F.date_trunc("month", "difficulty_date").cast("date"))
        .withColumn("chapter", F.substring("jasc", 1, 2))
        .withColumn("is_engine", F.col("chapter").isin(*ENGINE_CHAPTERS))
        .withColumn("stage_group", F.coalesce(stage_map[F.col("stage")], F.lit("Other / unknown")))
        .withColumn(
            "interruption",
            F.array_contains(interrupt, F.coalesce("precaution_a", F.lit("")))
            | F.array_contains(interrupt, F.coalesce("precaution_b", F.lit(""))),
        )
        .withColumn("text_len", F.length("text"))
        .withColumn(
            "action",
            F.when(F.upper("text").rlike(r"\b(R ?& ?R|R/R|REMOVED AND REPLACED|REPLACED|REPLACE)\b"), "Replace")
            .when(F.upper("text").rlike(r"\b(REPAIRED|REPAIR|REWORKED|REWORK|BLENDED|PATCHED)\b"), "Repair")
            .when(F.upper("text").rlike(r"\b(ADJUSTED|RIGGED|RESET|TIGHTENED|TORQUED|SECURED)\b"), "Adjust / secure")
            .when(F.upper("text").rlike(r"\b(CLEANED|FLUSHED|SERVICED|LUBRICATED)\b"), "Clean / service")
            .when(F.upper("text").rlike(r"\b(DEFERRED|MEL|INSPECTED|OPS CHECK|CHECKED GOOD|NO FAULT)\b"), "Inspect / defer")
            .otherwise("Unstated"),
        )
    )

    # Data-quality gates: reject rows that cannot be used downstream, count each reason.
    dq = sdr.agg(
        F.count("*").alias("rows"),
        F.sum(F.col("difficulty_date").isNull().cast("int")).alias("bad_date"),
        F.sum((~F.col("jasc").rlike(r"^\d{4}$") | F.col("jasc").isNull()).cast("int")).alias("bad_jasc"),
        F.sum((F.col("text").isNull() | (F.col("text_len") < 10)).cast("int")).alias("no_text"),
        F.sum((F.col("year") < 2016).cast("int")).alias("out_of_range_year"),
    ).first().asDict()

    clean = sdr.where(
        F.col("difficulty_date").isNotNull()
        & F.col("jasc").rlike(r"^\d{4}$")
        & (F.col("text_len") >= 10)
        & F.col("year").between(2016, 2025)
    )
    before_dedup = clean.count()
    clean = clean.dropDuplicates(["report_id", "jasc", "text"])
    n_clean = clean.count()
    dq.update(raw_rows=n_raw, duplicates=before_dedup - n_clean, clean_rows=n_clean)

    clean.repartition("year").write.mode("overwrite").partitionBy("year").parquet(str(PROCESSED / "sdr"))

    monthly = (
        clean.groupBy("month", "chapter")
        .agg(F.count("*").alias("reports"), F.sum(F.col("interruption").cast("int")).alias("interruptions"))
        .orderBy("month", "chapter")
    )
    monthly.write.mode("overwrite").parquet(str(PROCESSED / "sdr_monthly"))
    return dq


# ------------------------------------------------------------------------ C-MAPSS
def read_cmapss(spark: SparkSession, subset: str, split: str):
    path = RAW / "cmapss" / f"{split}_{subset}.txt"
    lines = spark.read.text(str(path))
    parts = F.split(F.trim(F.col("value")), r"\s+")
    cols = [parts.getItem(i).cast("double").alias(c) for i, c in enumerate(CMAPSS_COLS)]
    df = lines.select(*cols).withColumn("unit", F.col("unit").cast("int")).withColumn("cycle", F.col("cycle").cast("int"))
    return df.withColumn("subset", F.lit(subset)).withColumn("split", F.lit(split))


def etl_cmapss(spark: SparkSession) -> dict:
    frames = []
    for s in SUBSETS:
        for split in ("train", "test"):
            frames.append(read_cmapss(spark, s, split))
    df = frames[0]
    for f in frames[1:]:
        df = df.unionByName(f)

    # Ground-truth RUL for the test sets lives in a separate file, one value per unit.
    rul_rows = []
    for s in SUBSETS:
        vals = (RAW / "cmapss" / f"RUL_{s}.txt").read_text().split()
        rul_rows += [(s, i + 1, int(v)) for i, v in enumerate(vals)]
    rul_true = spark.createDataFrame(rul_rows, "subset string, unit int, rul_at_end int")

    w_engine = Window.partitionBy("subset", "split", "unit")
    df = (
        df.withColumn("engine_id", F.concat_ws("-", "subset", "split", F.format_string("%03d", "unit")))
        .withColumn("max_cycle", F.max("cycle").over(w_engine))
        .join(rul_true.withColumn("split", F.lit("test")), ["subset", "unit", "split"], "left")
        .withColumn("rul", (F.col("max_cycle") - F.col("cycle") + F.coalesce("rul_at_end", F.lit(0))).cast("int"))
        .withColumn("failed", (F.col("split") == "train") & (F.col("cycle") == F.col("max_cycle")))
    )

    # --- operating regimes: k-means on the three operational settings, fit on train only.
    assembler = VectorAssembler(inputCols=SETTINGS, outputCol="settings_vec")
    scaler = StandardScaler(inputCol="settings_vec", outputCol="settings_std", withMean=True)
    vec = assembler.transform(df)
    scaler_model = scaler.fit(vec.where(F.col("split") == "train"))
    vec = scaler_model.transform(vec)
    km = KMeans(k=6, seed=7, featuresCol="settings_std", predictionCol="regime_raw", maxIter=50)
    km_model = km.fit(vec.where(F.col("split") == "train"))
    df = km_model.transform(vec).drop("settings_vec", "settings_std")

    # Relabel clusters 0..5 by their mean altitude so the ids are stable and readable.
    order = (
        df.where(F.col("split") == "train").groupBy("regime_raw")
        .agg(F.avg("setting_1").alias("alt"), F.avg("setting_2").alias("mach"), F.avg("setting_3").alias("tra"), F.count("*").alias("n"))
        .orderBy("alt", "mach").collect()
    )
    relabel = {r["regime_raw"]: i for i, r in enumerate(order)}
    regime_table = [
        {"regime": relabel[r["regime_raw"]], "setting_1": r["alt"], "setting_2": r["mach"], "setting_3": r["tra"], "rows": r["n"]}
        for r in order
    ]
    mapping = F.create_map(*[F.lit(x) for kv in relabel.items() for x in kv])
    df = df.withColumn("regime", mapping[F.col("regime_raw")]).drop("regime_raw")

    # --- regime normalization: z-score each sensor within its regime (train statistics only).
    stats = df.where(F.col("split") == "train").groupBy("regime").agg(
        *[F.avg(s).alias(f"{s}_mu") for s in SENSORS], *[F.stddev(s).alias(f"{s}_sd") for s in SENSORS]
    )
    df = df.join(F.broadcast(stats), "regime", "left")
    for s in SENSORS:
        df = df.withColumn(
            f"{s}_n", F.when(F.col(f"{s}_sd") > 1e-6, (F.col(s) - F.col(f"{s}_mu")) / F.col(f"{s}_sd")).otherwise(F.lit(0.0))
        )
    df = df.drop(*[f"{s}_mu" for s in SENSORS], *[f"{s}_sd" for s in SENSORS])

    # --- rolling-window features per engine (trailing windows only: no look-ahead).
    w = Window.partitionBy("engine_id").orderBy("cycle")
    for s in INFORMATIVE:
        x = F.col(f"{s}_n")
        for win in (5, 15, 30):
            df = df.withColumn(f"{s}_m{win}", F.avg(x).over(w.rowsBetween(-(win - 1), 0)))
        df = df.withColumn(f"{s}_sd15", F.coalesce(F.stddev(x).over(w.rowsBetween(-14, 0)), F.lit(0.0)))
        # Least-squares slope over the trailing 30 cycles: cov(t, x) / var(t).
        w30 = w.rowsBetween(-29, 0)
        c = F.col("cycle").cast("double")
        cov = F.avg(c * x).over(w30) - F.avg(c).over(w30) * F.avg(x).over(w30)
        var = F.avg(c * c).over(w30) - F.avg(c).over(w30) ** 2
        df = df.withColumn(f"{s}_slope30", F.when(var > 0, cov / var).otherwise(F.lit(0.0)))

    df.repartition("subset").write.mode("overwrite").partitionBy("subset").parquet(str(PROCESSED / "cmapss_features"))

    summary = (
        df.groupBy("subset", "split")
        .agg(F.countDistinct("engine_id").alias("engines"), F.count("*").alias("rows"),
             F.avg("max_cycle").alias("mean_len"), F.countDistinct("regime").alias("regimes"))
        .orderBy("subset", "split").collect()
    )
    return {"regimes": regime_table, "summary": [r.asDict() for r in summary]}


def main() -> None:
    spark = spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    with Timer("SDR ETL") as t1:
        dq = etl_sdr(spark)
    print("  SDR data quality:", dq)
    with Timer("C-MAPSS ETL") as t2:
        cm = etl_cmapss(spark)
    for r in cm["summary"]:
        print("  ", r)
    export("etl", {
        "sdr_quality": dq,
        "cmapss": cm,
        "spark": {"version": spark.version, "master": spark.sparkContext.master,
                  "cores": spark.sparkContext.defaultParallelism,
                  "sdr_seconds": t1.seconds, "cmapss_seconds": t2.seconds},
    })
    spark.stop()


if __name__ == "__main__":
    main()
