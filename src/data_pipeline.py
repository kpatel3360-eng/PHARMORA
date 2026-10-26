"""
PHARMORA - Big Data Engineering Pipeline (PySpark & Spark SQL)
==============================================================
Production-grade data ingestion, validation, normalization, and analytical
aggregation pipeline for pharmaceutical sales intelligence.

Tech Stack: Apache Spark (PySpark), Spark SQL, Parquet
Author: Senior Big Data Engineer for PHARMORA MVP
"""

import os
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Dynamic Environment Configuration for Windows PySpark Execution
# ---------------------------------------------------------------------------
def _configure_spark_environment():
    """Configures JAVA_HOME, HADOOP_HOME and worker python interpreters to prevent Windows runtime issues."""
    project_root = Path(__file__).resolve().parent.parent

    # Configure JAVA_HOME if unset
    if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ.get("JAVA_HOME", "")):
        default_jdk = r"C:\Users\kpate\jdk-17\jdk-17.0.20.1+1"
        if os.path.exists(default_jdk):
            os.environ["JAVA_HOME"] = default_jdk
            os.environ["PATH"] = os.path.join(default_jdk, "bin") + os.pathsep + os.environ.get("PATH", "")

    # Configure HADOOP_HOME for Windows FileOutputCommitter Parquet write
    hadoop_dir = project_root / "hadoop"
    if hadoop_dir.exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        os.environ["PATH"] = str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")

    # Ensure Spark workers invoke the exact active Python interpreter
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

_configure_spark_environment()

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, IntegerType, DateType
    )
except ImportError as e:
    raise ImportError(
        f"PySpark is required but could not be imported: {e}\n"
        "Ensure PySpark is installed in your Python environment via:\n"
        "python -m pip install pyspark pyarrow\n"
    ) from e


# Medicine classification metadata (ATC Codes)
ATC_MEDICINE_METADATA = {
    "M01AB": "Anti-inflammatory (Acetic acid derivatives / Diclofenac)",
    "M01AE": "Anti-inflammatory (Propionic acid derivatives / Ibuprofen)",
    "N02BA": "Analgesics & Antipyretics (Salicylic acid / Aspirin)",
    "N02BE": "Analgesics & Antipyretics (Pyrazolones & Anilides / Paracetamol)",
    "N05B":  "Psycholeptics (Anxiolytic agents / Anti-anxiety)",
    "N05C":  "Psycholeptics (Hypnotics & Sedatives / Sleep aids)",
    "R03":   "Anti-asthmatic / Obstructive airway disease drugs",
    "R06":   "Antihistamines for systemic use (Allergy relief)"
}


def build_spark_session(app_name: str = "PharmoraSalesPipeline") -> SparkSession:
    """
    Initializes an optimized local SparkSession for data processing.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def run_pipeline(csv_path: str = "data/salesdaily.csv", output_dir: str = "processed") -> dict:
    """
    Executes the end-to-end PySpark pipeline:
    1. Loads raw CSV and inspects schema and row count.
    2. Parses dates and handles missing/invalid values.
    3. Unpivots wide medicine columns into standardized long format.
    4. Computes daily aggregations, total demand, and monthly trajectories.
    5. Executes analytical queries using Spark SQL.
    6. Persists processed datasets in optimized Parquet format.
    """
    input_file = Path(csv_path)
    output_path = Path(output_dir)

    print("=" * 75)
    print("  PHARMORA - BIG DATA ANALYTICS PIPELINE (PySpark & Spark SQL)")
    print("=" * 75)

    if not input_file.exists():
        raise FileNotFoundError(
            f"[ERROR] Raw dataset not found at '{input_file.resolve()}'.\n"
            f"Please verify that 'data/salesdaily.csv' exists before running."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    spark = build_spark_session()

    try:
        # -------------------------------------------------------------------
        # Step 1: Load CSV and Print Schema & Row Count
        # -------------------------------------------------------------------
        print(f"\n[STEP 1/6] Ingesting CSV dataset from: {input_file.as_posix()}")
        raw_df = spark.read.option("header", "true").csv(str(input_file))
        raw_count = raw_df.count()
        raw_columns = raw_df.columns

        print(f" -> Raw Records Ingested: {raw_count:,}")
        print(f" -> Columns Detected ({len(raw_columns)}): {raw_columns}")
        print("\n--- Ingested Schema ---")
        raw_df.printSchema()

        # -------------------------------------------------------------------
        # Step 2: Validate and Parse Dates
        # -------------------------------------------------------------------
        print("[STEP 2/6] Cleaning and parsing date formats...")
        
        # Real dates in salesdaily.csv are in M/d/yyyy format (e.g. 1/2/2014)
        df_with_date = raw_df.withColumn(
            "parsed_date",
            F.coalesce(
                F.to_date(F.col("datum"), "M/d/yyyy"),
                F.to_date(F.col("datum"), "MM/dd/yyyy"),
                F.to_date(F.col("datum"), "yyyy-MM-dd"),
                F.to_date(F.col("datum"), "d/M/yyyy")
            )
        )

        null_dates_count = df_with_date.filter(F.col("parsed_date").isNull()).count()
        print(f" -> Null or unparseable dates detected: {null_dates_count}")

        if null_dates_count > 0:
            print(f" -> [NOTICE] Filtering out {null_dates_count} records with invalid dates.")
            df_valid_dates = df_with_date.filter(F.col("parsed_date").isNotNull())
        else:
            df_valid_dates = df_with_date

        # -------------------------------------------------------------------
        # Step 3: Unpivot Wide-Format Medicine Columns to Long Format
        # -------------------------------------------------------------------
        print("[STEP 3/6] Transforming wide medicine columns to normalized long format...")
        
        # Identify medicine columns present in the dataset
        medicine_cols = [c for c in raw_columns if c in ATC_MEDICINE_METADATA]
        if not medicine_cols:
            # Fallback to any non-metadata column
            exclude = {"datum", "Year", "Month", "Hour", "Weekday Name", "parsed_date"}
            medicine_cols = [c for c in raw_columns if c not in exclude]

        print(f" -> Target Medicine Categories ({len(medicine_cols)}): {medicine_cols}")

        # Construct dynamic SQL stack expression for unpivoting
        # stack(n, 'col1', double(col1), 'col2', double(col2), ...)
        stack_parts = []
        for m in medicine_cols:
            stack_parts.append(f"'{m}', coalesce(double(`{m}`), 0.0)")
        stack_expr = f"stack({len(medicine_cols)}, {', '.join(stack_parts)}) as (medicine_id, units_sold)"

        df_long = df_valid_dates.select(
            F.col("parsed_date").alias("date"),
            F.expr(stack_expr)
        )

        # Sanitize numeric values: sales units cannot be negative
        df_cleaned = df_long.filter(F.col("units_sold") >= 0.0)

        # Check if persistent staging data/imported_sales.csv exists
        imported_staging = Path("data/imported_sales.csv")
        if imported_staging.exists() and imported_staging.stat().st_size > 0:
            print(f" -> [IMPORT] Staged import records detected at '{imported_staging.as_posix()}'. Ingesting into pipeline...")
            raw_imported = spark.read.option("header", "true").csv(str(imported_staging))
            if "date" in raw_imported.columns and "medicine_id" in raw_imported.columns and "units_sold" in raw_imported.columns:
                imported_clean = raw_imported.select(
                    F.coalesce(
                        F.to_date(F.col("date"), "yyyy-MM-dd"),
                        F.to_date(F.col("date"), "M/d/yyyy"),
                        F.to_date(F.col("date"), "MM/dd/yyyy")
                    ).alias("date"),
                    F.trim(F.upper(F.col("medicine_id"))).alias("medicine_id"),
                    F.coalesce(F.col("units_sold").cast(DoubleType()), F.lit(0.0)).alias("units_sold")
                ).filter(F.col("date").isNotNull() & (F.col("units_sold") >= 0.0))
                df_cleaned = df_cleaned.unionByName(imported_clean).dropDuplicates(["date", "medicine_id"])

        # Enrich with calendar dimensions
        df_enriched = df_cleaned.withColumn("year", F.year("date")) \
                                .withColumn("month", F.month("date")) \
                                .withColumn("day_of_month", F.dayofmonth("date")) \
                                .withColumn("day_of_week", F.date_format("date", "EEEE"))

        total_long_records = df_enriched.count()
        print(f" -> Normalized Long-Format Record Count: {total_long_records:,}")

        # -------------------------------------------------------------------
        # Step 4: Spark SQL Analytical Engine Queries
        # -------------------------------------------------------------------
        print("\n[STEP 4/6] Registering Spark SQL Views and running analytical queries...")
        df_enriched.createOrReplaceTempView("daily_pharma_sales")

        # Spark SQL Query 1: Medicine category performance summary
        print("\n--- [Spark SQL Query 1] Overall Medicine Performance Metrics ---")
        sql_summary = spark.sql("""
            SELECT 
                medicine_id,
                COUNT(date) AS total_active_days,
                ROUND(SUM(units_sold), 2) AS total_units_sold,
                ROUND(AVG(units_sold), 2) AS avg_daily_demand,
                ROUND(STDDEV(units_sold), 2) AS demand_volatility,
                ROUND(MAX(units_sold), 2) AS peak_single_day_sales
            FROM daily_pharma_sales
            GROUP BY medicine_id
            ORDER BY total_units_sold DESC
        """)
        sql_summary.show(truncate=False)

        # Spark SQL Query 2: Day-of-Week Purchasing Patterns
        print("--- [Spark SQL Query 2] Purchasing Patterns by Weekday ---")
        sql_weekday = spark.sql("""
            SELECT 
                day_of_week,
                ROUND(SUM(units_sold), 2) AS total_sales,
                ROUND(AVG(units_sold), 2) AS avg_sales_per_record
            FROM daily_pharma_sales
            GROUP BY day_of_week
            ORDER BY total_sales DESC
        """)
        sql_weekday.show(truncate=False)

        # -------------------------------------------------------------------
        # Step 5: Aggregations (Daily Sales, Medicine Summary, Monthly Trends)
        # -------------------------------------------------------------------
        print("[STEP 5/6] Aggregating sales metrics and monthly trends...")
        
        # Aggregation A: Monthly trends
        df_monthly_trends = spark.sql("""
            SELECT 
                year,
                month,
                medicine_id,
                ROUND(SUM(units_sold), 2) AS monthly_units_sold,
                ROUND(AVG(units_sold), 2) AS avg_daily_sales,
                COUNT(date) AS days_recorded
            FROM daily_pharma_sales
            GROUP BY year, month, medicine_id
            ORDER BY year, month, medicine_id
        """)

        # -------------------------------------------------------------------
        # Step 6: Persist Processed Datasets as Parquet
        # -------------------------------------------------------------------
        print("[STEP 6/6] Persisting processed datasets to Parquet in 'processed/'...")
        
        daily_parquet_path = output_path / "daily_sales.parquet"
        summary_parquet_path = output_path / "medicine_summary.parquet"
        monthly_parquet_path = output_path / "monthly_trends.parquet"

        # Write clean normalized daily sales
        df_enriched.write.mode("overwrite").parquet(str(daily_parquet_path))
        print(f" -> Saved Daily Sales Parquet: {daily_parquet_path.as_posix()}")

        # Write analytical summaries
        sql_summary.write.mode("overwrite").parquet(str(summary_parquet_path))
        print(f" -> Saved Medicine Summary Parquet: {summary_parquet_path.as_posix()}")

        # Write monthly trends
        df_monthly_trends.write.mode("overwrite").parquet(str(monthly_parquet_path))
        print(f" -> Saved Monthly Trends Parquet: {monthly_parquet_path.as_posix()}")

        summary_results = {
            "raw_records": raw_count,
            "normalized_records": total_long_records,
            "medicine_count": len(medicine_cols),
            "parquet_files": [
                str(daily_parquet_path),
                str(summary_parquet_path),
                str(monthly_parquet_path)
            ]
        }

        print("\n" + "=" * 75)
        print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print(f"  Processed {raw_count:,} raw daily rows into {total_long_records:,} long-format facts.")
        print(f"  Parquet artifacts generated in '{output_dir}/'.")
        print("=" * 75)
        return summary_results

    finally:
        # Graceful shutdown of SparkSession
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description="PHARMORA PySpark Data Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="data/salesdaily.csv",
        help="Path to input sales CSV file (default: data/salesdaily.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed",
        help="Path to output directory for Parquet files (default: processed)"
    )
    args = parser.parse_args()
    run_pipeline(csv_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
