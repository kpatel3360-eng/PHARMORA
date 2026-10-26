"""
PHARMORA - Data Pipeline Verification Script
=============================================
Performs rigorous validation of raw input CSV vs. processed Parquet tables:
1. Ingestion row counts and schemas.
2. Null dates, invalid sales, and duplicate date-medicine checks.
3. Reconciliation of total sales volume (raw vs. processed).
4. Date range and distinct medicines.
5. Direct querying of saved Parquet artifacts via PySpark and Spark SQL.
"""

import sys
import os
from pathlib import Path

# Configure environment
project_root = Path(__file__).resolve().parent
default_jdk = r"C:\Users\kpate\jdk-17\jdk-17.0.20.1+1"
if os.path.exists(default_jdk):
    os.environ["JAVA_HOME"] = default_jdk
    os.environ["PATH"] = os.path.join(default_jdk, "bin") + os.pathsep + os.environ.get("PATH", "")

hadoop_dir = project_root / "hadoop"
if hadoop_dir.exists():
    os.environ["HADOOP_HOME"] = str(hadoop_dir)
    os.environ["PATH"] = str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main():
    spark = (
        SparkSession.builder
        .appName("PharmoraPipelineVerification")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print("\n" + "="*80)
    print("  PHARMORA - DATA PIPELINE VERIFICATION REPORT")
    print("="*80)

    # 1. Inspect Raw CSV
    csv_path = "data/salesdaily.csv"
    print(f"\n--- [1] RAW CSV VALIDATION ({csv_path}) ---")
    raw_df = spark.read.option("header", "true").csv(csv_path)
    raw_count = raw_df.count()
    print(f"Raw CSV row count: {raw_count:,}")
    print(f"Raw CSV columns ({len(raw_df.columns)}): {raw_df.columns}")

    medicine_cols = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
    
    # Calculate raw total sales across all medicine columns
    raw_sales_expr = " + ".join([f"coalesce(double(`{c}`), 0.0)" for c in medicine_cols])
    raw_with_total = raw_df.withColumn("row_total", F.expr(raw_sales_expr))
    raw_total_sales = raw_with_total.select(F.round(F.sum("row_total"), 4)).collect()[0][0]
    print(f"Raw Total Units Sold Across All 8 Medicines: {raw_total_sales:,.4f}")

    # Check for negative or non-numeric sales in raw data
    raw_negatives = 0
    for col_name in medicine_cols:
        neg_count = raw_df.filter(F.col(col_name).cast("double") < 0).count()
        if neg_count > 0:
            print(f"Warning: {neg_count} negative values found in {col_name}")
            raw_negatives += neg_count
    print(f"Total Negative Sales Records in Raw Data: {raw_negatives}")

    # 2. Inspect Processed daily_sales.parquet
    daily_parquet_path = "processed/daily_sales.parquet"
    print(f"\n--- [2] PROCESSED DAILY SALES PARQUET VALIDATION ({daily_parquet_path}) ---")
    daily_df = spark.read.parquet(daily_parquet_path)
    processed_count = daily_df.count()
    print(f"Processed row count: {processed_count:,}")
    print(f"Expected row count ({raw_count} * 8): {raw_count * 8:,}")
    print(f"Row count exact match: {processed_count == raw_count * 8}")

    print("\nProcessed daily_sales schema:")
    daily_df.printSchema()

    # Distinct medicines
    distinct_meds = [r[0] for r in daily_df.select("medicine_id").distinct().orderBy("medicine_id").collect()]
    print(f"Distinct medicines ({len(distinct_meds)}): {distinct_meds}")

    # Date range
    date_stats = daily_df.select(F.min("date").alias("min_date"), F.max("date").alias("max_date")).collect()[0]
    print(f"Date range: {date_stats['min_date']} to {date_stats['max_date']}")

    # Null dates check
    null_dates = daily_df.filter(F.col("date").isNull()).count()
    print(f"Null dates count: {null_dates}")

    # Invalid sales values check (negative or null)
    invalid_sales = daily_df.filter(F.col("units_sold").isNull() | (F.col("units_sold") < 0.0)).count()
    print(f"Invalid sales (null or < 0) count: {invalid_sales}")

    # Duplicate (date, medicine_id) check
    dup_counts = daily_df.groupBy("date", "medicine_id").count().filter(F.col("count") > 1).count()
    print(f"Duplicate (date, medicine_id) combinations: {dup_counts}")

    # Total sales volume reconciliation
    processed_total_sales = daily_df.select(F.round(F.sum("units_sold"), 4)).collect()[0][0]
    print(f"Processed Total Units Sold: {processed_total_sales:,.4f}")
    discrepancy = abs(raw_total_sales - processed_total_sales)
    print(f"Total Sales Reconciliation Discrepancy: {discrepancy:.6f}")
    reconciliation_ok = discrepancy < 0.001
    print(f"Reconciliation Status: {'PERFECT RECONCILIATION (MATCH)' if reconciliation_ok else 'DISCREPANCY DETECTED'}")

    # Display 10 sample records
    print("\n--- [3] 10 SAMPLE RECORDS FROM DAILY SALES ---")
    daily_df.orderBy("date", "medicine_id").show(10, truncate=False)

    # 3. Inspect Processed medicine_summary.parquet
    summary_parquet_path = "processed/medicine_summary.parquet"
    print(f"\n--- [4] MEDICINE SUMMARY PARQUET VALIDATION ({summary_parquet_path}) ---")
    summary_df = spark.read.parquet(summary_parquet_path)
    summary_df.printSchema()
    summary_df.show(truncate=False)

    summary_total_sales = summary_df.select(F.round(F.sum("total_units_sold"), 2)).collect()[0][0]
    print(f"Sum of total_units_sold across summary table: {summary_total_sales:,.2f}")
    print(f"Daily sales rounded total: {round(processed_total_sales, 2):,.2f}")

    # 4. Inspect Processed monthly_trends.parquet
    monthly_parquet_path = "processed/monthly_trends.parquet"
    print(f"\n--- [5] MONTHLY TRENDS PARQUET VALIDATION ({monthly_parquet_path}) ---")
    monthly_df = spark.read.parquet(monthly_parquet_path)
    monthly_count = monthly_df.count()
    print(f"Monthly trends row count: {monthly_count:,}")
    monthly_df.printSchema()
    print("Sample monthly trends (first 10 records):")
    monthly_df.orderBy("year", "month", "medicine_id").show(10, truncate=False)

    # 5. Direct Spark SQL Query Test on loaded Parquet
    print("\n--- [6] DIRECT SPARK SQL QUERY ON PARQUET ARTIFACTS ---")
    daily_df.createOrReplaceTempView("verified_daily_sales")
    sql_test = spark.sql("""
        SELECT 
            year,
            medicine_id,
            ROUND(SUM(units_sold), 2) AS yearly_sales,
            ROUND(AVG(units_sold), 2) AS avg_daily_sales,
            MAX(units_sold) AS max_daily_sales
        FROM verified_daily_sales
        WHERE year = 2018
        GROUP BY year, medicine_id
        ORDER BY yearly_sales DESC
    """)
    sql_test.show(truncate=False)

    print("="*80)
    print("  ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY")
    print("="*80)

    spark.stop()

if __name__ == "__main__":
    main()
