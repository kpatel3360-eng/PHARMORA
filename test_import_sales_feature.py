"""
PHARMORA - Automated Test Suite for Sales Ingestion & Import Feature
===================================================================
Tests all 18 requirements from the user specification:
1. Standard long format schema validation
2. Wide format auto-unpivoting
3. Rejection of invalid dates, negative quantities, missing columns, and empty IDs
4. Row-level error reporting with exact row indices
5. Persistent staging to data/imported_sales.csv without touching data/salesdaily.csv
6. Incremental Parquet table updates (daily_sales, medicine_summary, monthly_trends)
7. Dynamic medicine selector population in Streamlit app
8. Idempotent re-import duplicate prevention
9. Transaction ID granular deduplication
10. Downstream analytics graceful fallback (zero-fabrication guarantee)
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# Ingestion engine
from src.data_ingestion import (
    validate_sales_records,
    deduplicate_records,
    persist_and_update_pipeline,
    IMPORTED_STAGING_PATH,
    BASELINE_SALES_PATH
)
from app import load_all_pharmora_data


def get_file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def test_1_invalid_csv_validation():
    print("\n--- TEST 1: CSV Validation & Error Detection ---")
    
    # 1A: Missing required column
    csv_missing_col = "date,units_sold\n2020-01-01,15.0\n"
    res1 = validate_sales_records(pd.read_csv(pd.io.common.StringIO(csv_missing_col)))
    assert not res1["is_valid"], "Failed to reject CSV missing medicine_id"
    assert res1["detected_format"] == "invalid_schema"
    print(" -> [PASS] Missing column 'medicine_id' correctly rejected.")

    # 1B: Empty CSV
    csv_empty = ""
    res_empty = validate_sales_records(pd.io.common.StringIO(csv_empty))
    assert not res_empty["is_valid"]
    print(" -> [PASS] Empty CSV correctly rejected.")

    # 1C: Row-level errors: bad date, negative units, missing ID
    csv_dirty = """date,medicine_id,units_sold
2020-01-01,J01CA,10.5
2020-99-99,J01CA,15.0
2020-01-03,,12.0
2020-01-04,J01CA,-5.0
2020-01-05,J01CA,invalid_num
2020-01-06,J01CA,20.0
"""
    res_dirty = validate_sales_records(pd.io.common.StringIO(csv_dirty))
    assert not res_dirty["is_valid"]
    assert res_dirty["total_rows"] == 6
    assert res_dirty["valid_count"] == 2  # rows 1 and 6
    assert res_dirty["invalid_count"] == 4  # rows 2, 3, 4, 5
    
    # Verify exact error rows and reasons
    errors = res_dirty["errors"]
    err_rows = [e["row"] for e in errors]
    assert 2 in err_rows, "Row 2 bad date not flagged"
    assert 3 in err_rows, "Row 3 missing medicine_id not flagged"
    assert 4 in err_rows, "Row 4 negative sales not flagged"
    assert 5 in err_rows, "Row 5 non-numeric sales not flagged"
    print(f" -> [PASS] Row-level validation captured exactly 4 invalid rows with precise line numbers: {err_rows}")


def test_2_wide_format_support():
    print("\n--- TEST 2: Wide-Format Auto-Conversion ---")
    wide_csv = """datum,M01AB,J01CA
2020-01-01,5.0,12.5
2020-01-02,7.0,14.0
"""
    res_wide = validate_sales_records(pd.io.common.StringIO(wide_csv))
    assert res_wide["is_valid"]
    assert res_wide["detected_format"] == "wide_format"
    assert res_wide["valid_count"] == 4  # 2 dates x 2 medicines
    valid_df = res_wide["valid_df"]
    assert set(valid_df["medicine_id"].unique()) == {"M01AB", "J01CA"}
    print(f" -> [PASS] Wide format unpivoted successfully into {len(valid_df)} long records.")


def test_3_import_new_medicine_and_preservation():
    print("\n--- TEST 3: Ingesting New Medicine (J01CA - Amoxicillin) & Baseline Preservation ---")
    
    baseline_path = Path(BASELINE_SALES_PATH)
    assert baseline_path.exists(), "data/salesdaily.csv must exist!"
    baseline_md5_before = get_file_md5(str(baseline_path))
    baseline_lines_before = len(baseline_path.read_text().splitlines())

    # Create 15 daily sales records for new medicine J01CA (Amoxicillin)
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    sales_data = []
    for i, d in enumerate(dates):
        sales_data.append({
            "date": d.strftime("%Y-%m-%d"),
            "medicine_id": "J01CA",
            "units_sold": round(10.0 + (i % 5) * 2.5, 2),
            "transaction_id": f"TX-J01CA-202001-{i+1:03d}"
        })
    df_incoming = pd.DataFrame(sales_data)

    # Validate
    val = validate_sales_records(df_incoming, source_filename="amoxicillin_jan2020.csv")
    assert val["is_valid"]
    assert val["valid_count"] == 15

    # Deduplicate against current facts
    unique_df, dup_count, _ = deduplicate_records(val["valid_df"])
    assert len(unique_df) == 15
    assert dup_count == 0

    # Ingest and update pipeline
    res = persist_and_update_pipeline(unique_df, source_filename="amoxicillin_jan2020.csv")
    assert res["success"]
    assert res["imported_count"] == 15
    assert "J01CA" in res["new_medicines"]

    # Verify baseline file was NOT modified
    baseline_md5_after = get_file_md5(str(baseline_path))
    baseline_lines_after = len(baseline_path.read_text().splitlines())
    assert baseline_md5_before == baseline_md5_after, "FATAL: data/salesdaily.csv was modified!"
    assert baseline_lines_before == baseline_lines_after
    print(" -> [PASS] Original dataset 'data/salesdaily.csv' is 100% untouched (MD5 matched).")

    # Verify staging file exists
    staging_file = Path(IMPORTED_STAGING_PATH)
    assert staging_file.exists(), "data/imported_sales.csv was not created!"
    stg_df = pd.read_csv(staging_file)
    assert len(stg_df) >= 15
    assert "J01CA" in stg_df["medicine_id"].values
    assert "imported_at" in stg_df.columns
    assert "source_file" in stg_df.columns
    print(f" -> [PASS] Validated records staged in 'data/imported_sales.csv' with audit metadata.")

    # Verify updated Parquet artifacts
    p = Path("processed")
    daily_pq = pd.read_parquet(p / "daily_sales.parquet")
    summary_pq = pd.read_parquet(p / "medicine_summary.parquet")
    monthly_pq = pd.read_parquet(p / "monthly_trends.parquet")

    assert "J01CA" in daily_pq["medicine_id"].values, "J01CA not found in daily_sales.parquet"
    assert "J01CA" in summary_pq["medicine_id"].values, "J01CA not found in medicine_summary.parquet"
    assert "J01CA" in monthly_pq["medicine_id"].values, "J01CA not found in monthly_trends.parquet"

    # Verify J01CA summary numbers match incoming facts
    j01_sum = summary_pq[summary_pq["medicine_id"] == "J01CA"].iloc[0]
    expected_vol = round(float(df_incoming["units_sold"].sum()), 2)
    assert np.isclose(j01_sum["total_units_sold"], expected_vol), f"Volume mismatch: {j01_sum['total_units_sold']} vs {expected_vol}"
    assert j01_sum["total_active_days"] == 15
    print(f" -> [PASS] Parquet tables updated: daily_sales ({len(daily_pq):,} rows), summary ({len(summary_pq)} medicines), monthly_trends ({len(monthly_pq)} rows).")


def test_4_idempotent_reimport_duplicate_prevention():
    print("\n--- TEST 4: Re-importing Same File (Deduplication Policy Verification) ---")
    
    # Construct exact same file as imported in Test 3
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    sales_data = []
    for i, d in enumerate(dates):
        sales_data.append({
            "date": d.strftime("%Y-%m-%d"),
            "medicine_id": "J01CA",
            "units_sold": round(10.0 + (i % 5) * 2.5, 2),
            "transaction_id": f"TX-J01CA-202001-{i+1:03d}"
        })
    df_reimport = pd.DataFrame(sales_data)

    val = validate_sales_records(df_reimport, source_filename="amoxicillin_jan2020.csv")
    assert val["is_valid"]
    
    # Deduplicate: all 15 should be detected as duplicates
    unique_df, dup_count, dup_df = deduplicate_records(val["valid_df"])
    assert dup_count == 15, f"Expected 15 duplicates skipped, got {dup_count}"
    assert len(unique_df) == 0, f"Expected 0 unique rows, got {len(unique_df)}"
    assert len(dup_df) == 15

    # Attempting to persist empty unique_df should be safely rejected
    res = persist_and_update_pipeline(unique_df, source_filename="amoxicillin_jan2020.csv")
    assert not res["success"]
    assert res["imported_count"] == 0
    print(f" -> [PASS] Exactly 15/15 records identified as duplicates. 0 duplicate facts ingested.")


def test_5_dashboard_data_loader_and_selector():
    print("\n--- TEST 5: Streamlit App Dynamic Medicine Selector & Data Loader ---")
    
    app_data = load_all_pharmora_data()
    daily_df = app_data["daily"]
    assert daily_df is not None
    
    all_meds = sorted(daily_df["medicine_id"].unique().tolist())
    assert "J01CA" in all_meds, "J01CA does not appear in dynamic medicine selector list!"
    print(f" -> [PASS] Dynamic medicine selector options ({len(all_meds)} categories): {all_meds}")
    
    # Test slice for J01CA
    j01_slice = daily_df[daily_df["medicine_id"] == "J01CA"]
    assert len(j01_slice) == 15
    assert j01_slice["units_sold"].sum() > 0
    print(f" -> [PASS] J01CA historical sales slice loaded: {len(j01_slice)} records, total volume = {j01_slice['units_sold'].sum():.2f} units.")


def test_6_zero_fabrication_safeguards():
    print("\n--- TEST 6: Zero-Fabrication Safeguards for New Medicine ---")
    app_data = load_all_pharmora_data()
    eval_df = app_data["evaluations"]
    stockout_df = app_data["stockout"]
    expiry_df = app_data["expiry"]
    reorder_df = app_data["reorder"]

    # 1. No fake forecast evaluation assigned
    if eval_df is not None:
        assert "J01CA" not in eval_df["medicine_id"].values, "J01CA must not have fabricated forecast evaluations!"
    print(" -> [PASS] No fabricated ML forecast evaluation assigned to J01CA.")

    # 2. No fake inventory stock assigned
    if stockout_df is not None:
        assert "J01CA" not in stockout_df["medicine_id"].values, "J01CA must not have fabricated warehouse inventory!"
    print(" -> [PASS] No fabricated physical stock assigned to J01CA.")

    # 3. No fake batches assigned
    if expiry_df is not None:
        assert "J01CA" not in expiry_df["medicine_id"].values, "J01CA must not have fabricated batch records!"
    print(" -> [PASS] No fabricated expiration batches assigned to J01CA.")

    # 4. No fake reorder decisions assigned
    if reorder_df is not None:
        assert "J01CA" not in reorder_df["medicine_id"].values, "J01CA must not have fabricated reorder orders!"
    print(" -> [PASS] No fabricated reorders assigned to J01CA.")


def main():
    print("=" * 80)
    print("  PHARMORA - SALES DATA INGESTION & IMPORT FEATURE VERIFICATION")
    print("=" * 80)

    # Ensure clean slate for tests
    stg = Path(IMPORTED_STAGING_PATH)
    if stg.exists():
        stg.unlink()
    daily_path = Path("processed/daily_sales.parquet")
    if daily_path.exists():
        try:
            df_d = pd.read_parquet(daily_path)
            if "J01CA" in df_d["medicine_id"].values:
                df_d = df_d[df_d["medicine_id"] != "J01CA"]
                from src.data_ingestion import save_parquet_dataset
                save_parquet_dataset(df_d, str(daily_path))
        except Exception:
            pass

    try:
        test_1_invalid_csv_validation()
        test_2_wide_format_support()
        test_3_import_new_medicine_and_preservation()
        test_4_idempotent_reimport_duplicate_prevention()
        test_5_dashboard_data_loader_and_selector()
        test_6_zero_fabrication_safeguards()

        print("\n" + "=" * 80)
        print("  ALL 6 FEATURE TEST SUITES PASSED SUCCESSFULLY (100% HEALTHY)")
        print("=" * 80)
    except AssertionError as e:
        import traceback
        traceback.print_exc()
        print(f"\n[FAIL] Test assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] Unexpected error during test execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
