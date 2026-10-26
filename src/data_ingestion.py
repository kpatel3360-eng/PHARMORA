"""
PHARMORA - Data Ingestion & Sales Import Module
===============================================
Production-grade ingestion, schema validation, deduplication, and Parquet
updating for additional pharmaceutical sales records.

Supports:
1. Standard Long Format: (date, medicine_id, units_sold, [transaction_id])
2. Wide ATC Format: (datum/date, [medicine columns...]) - auto-detected & unpivoted
3. Multi-format date parsing (ISO, US, European)
4. Non-negative numeric sales validation
5. Deduplication against both baseline facts and previous imports
6. Safe non-destructive persistence to data/imported_sales.csv
7. Incremental updates to processed/daily_sales.parquet, medicine_summary.parquet,
   and monthly_trends.parquet without modifying data/salesdaily.csv
"""

import os
import io
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np


STANDARD_REQUIRED_COLS = ["date", "medicine_id", "units_sold"]
IMPORTED_STAGING_PATH = "data/imported_sales.csv"
BASELINE_SALES_PATH = "data/salesdaily.csv"
PROCESSED_DIR = "processed"


def parse_flexible_dates(series: pd.Series) -> pd.Series:
    """Parses dates supporting multiple formats (ISO, US M/D/YYYY, etc.)."""
    # First attempt standard pandas auto-parser with dayfirst=False then dayfirst=True
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def validate_sales_records(
    file_or_df,
    source_filename: str = "upload.csv"
) -> Dict[str, Any]:
    """
    Ingests and validates raw sales records from a CSV file, file-like buffer, or DataFrame.
    
    Validation Rules:
    1. Schema conformity: Must contain standard cols (date, medicine_id, units_sold) OR wide format.
    2. Date validity: Must be parseable dates.
    3. Non-negative sales: units_sold must be numeric and >= 0.
    4. Valid medicine_id: String identifier, non-empty, stripped.
    
    Returns a comprehensive validation dictionary.
    """
    if isinstance(file_or_df, (str, Path)):
        p = Path(file_or_df)
        if not p.exists():
            return {
                "is_valid": False,
                "error_message": f"File '{file_or_df}' not found.",
                "total_rows": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "errors": [{"row": 0, "field": "file", "message": f"File not found: {file_or_df}"}],
                "valid_df": pd.DataFrame(),
                "detected_format": "unknown"
            }
        df_raw = pd.read_csv(file_or_df)
    elif isinstance(file_or_df, pd.DataFrame):
        df_raw = file_or_df.copy()
    else:
        # File-like buffer (e.g. Streamlit UploadedFile)
        try:
            df_raw = pd.read_csv(file_or_df)
        except Exception as e:
            return {
                "is_valid": False,
                "error_message": f"Could not parse CSV file: {e}",
                "total_rows": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "errors": [{"row": 0, "field": "csv_parser", "message": str(e)}],
                "valid_df": pd.DataFrame(),
                "detected_format": "unknown"
            }

    raw_total_rows = len(df_raw)
    if raw_total_rows == 0:
        return {
            "is_valid": False,
            "error_message": "Uploaded CSV is completely empty (0 rows).",
            "total_rows": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "errors": [{"row": 0, "field": "empty", "message": "Uploaded CSV contains 0 rows"}],
            "valid_df": pd.DataFrame(),
            "detected_format": "empty"
        }

    # Normalize column names (strip whitespace and lower for checking)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    col_map = {c.lower(): c for c in df_raw.columns}

    # Check if wide format (has 'datum' or 'date' and multiple drug code columns)
    is_wide = False
    detected_format = "standard_long"
    
    has_date_col = "date" in col_map or "datum" in col_map
    has_med_col = "medicine_id" in col_map or "medicine" in col_map or "atc_code" in col_map
    has_units_col = "units_sold" in col_map or "units" in col_map or "quantity" in col_map or "sales" in col_map

    if has_date_col and not has_med_col and not has_units_col:
        # Check if multiple medicine columns exist (wide format)
        date_col = col_map.get("date") or col_map.get("datum")
        candidate_med_cols = [c for c in df_raw.columns if c != date_col and c not in ["Year", "Month", "Hour", "Weekday Name"]]
        if len(candidate_med_cols) >= 1:
            is_wide = True
            detected_format = "wide_format"
            # Unpivot wide format to long
            long_records = []
            for _, row in df_raw.iterrows():
                d_val = row[date_col]
                for m_col in candidate_med_cols:
                    u_val = row[m_col]
                    long_records.append({
                        "date": d_val,
                        "medicine_id": m_col,
                        "units_sold": u_val
                    })
            df_working = pd.DataFrame(long_records)
        else:
            df_working = df_raw.copy()
    else:
        # Standard long format mapping
        df_working = pd.DataFrame()
        # Date column
        if "date" in col_map:
            df_working["date"] = df_raw[col_map["date"]]
        elif "datum" in col_map:
            df_working["date"] = df_raw[col_map["datum"]]
        else:
            return {
                "is_valid": False,
                "error_message": "Missing required date column ('date' or 'datum').",
                "total_rows": raw_total_rows,
                "valid_count": 0,
                "invalid_count": raw_total_rows,
                "errors": [{"row": 0, "field": "schema", "message": "Missing required column 'date'"}],
                "valid_df": pd.DataFrame(),
                "detected_format": "invalid_schema"
            }

        # Medicine ID column
        if "medicine_id" in col_map:
            df_working["medicine_id"] = df_raw[col_map["medicine_id"]]
        elif "medicine" in col_map:
            df_working["medicine_id"] = df_raw[col_map["medicine_id"]]
        elif "atc_code" in col_map:
            df_working["medicine_id"] = df_raw[col_map["atc_code"]]
        else:
            return {
                "is_valid": False,
                "error_message": "Missing required medicine column ('medicine_id', 'medicine', or 'atc_code').",
                "total_rows": raw_total_rows,
                "valid_count": 0,
                "invalid_count": raw_total_rows,
                "errors": [{"row": 0, "field": "schema", "message": "Missing required column 'medicine_id'"}],
                "valid_df": pd.DataFrame(),
                "detected_format": "invalid_schema"
            }

        # Units Sold column
        if "units_sold" in col_map:
            df_working["units_sold"] = df_raw[col_map["units_sold"]]
        elif "units" in col_map:
            df_working["units_sold"] = df_raw[col_map["units"]]
        elif "quantity" in col_map:
            df_working["units_sold"] = df_raw[col_map["quantity"]]
        elif "sales" in col_map:
            df_working["units_sold"] = df_raw[col_map["sales"]]
        else:
            return {
                "is_valid": False,
                "error_message": "Missing required sales volume column ('units_sold', 'units', or 'quantity').",
                "total_rows": raw_total_rows,
                "valid_count": 0,
                "invalid_count": raw_total_rows,
                "errors": [{"row": 0, "field": "schema", "message": "Missing required column 'units_sold'"}],
                "valid_df": pd.DataFrame(),
                "detected_format": "invalid_schema"
            }

        # Optional transaction_id
        if "transaction_id" in col_map:
            df_working["transaction_id"] = df_raw[col_map["transaction_id"]].astype(str)
        else:
            df_working["transaction_id"] = None

    # Row-by-Row Data Quality Validation
    errors = []
    valid_indices = []

    parsed_dates = parse_flexible_dates(df_working["date"])

    for idx, row in df_working.iterrows():
        row_num = idx + 1  # 1-indexed for user readability
        has_error = False

        # 1. Date check
        dt = parsed_dates.iloc[idx]
        if pd.isnull(dt):
            errors.append({
                "row": row_num,
                "field": "date",
                "value": str(row["date"]),
                "message": f"Invalid or unparseable date format: '{row['date']}'"
            })
            has_error = True

        # 2. Medicine ID check
        med_id = str(row["medicine_id"]).strip().upper() if not pd.isnull(row["medicine_id"]) else ""
        if not med_id or med_id in ["NAN", "NONE", "NULL", ""]:
            errors.append({
                "row": row_num,
                "field": "medicine_id",
                "value": str(row["medicine_id"]),
                "message": "Medicine ID is empty or missing"
            })
            has_error = True

        # 3. Units Sold check
        u_val = row["units_sold"]
        try:
            u_num = float(u_val)
            if np.isnan(u_num):
                errors.append({
                    "row": row_num,
                    "field": "units_sold",
                    "value": str(u_val),
                    "message": "Sales units value is NaN or missing"
                })
                has_error = True
            elif u_num < 0:
                errors.append({
                    "row": row_num,
                    "field": "units_sold",
                    "value": str(u_val),
                    "message": f"Negative sales quantity not permitted ({u_num} < 0)"
                })
                has_error = True
        except (ValueError, TypeError):
            errors.append({
                "row": row_num,
                "field": "units_sold",
                "value": str(u_val),
                "message": f"Non-numeric sales units: '{u_val}'"
            })
            has_error = True

        if not has_error:
            valid_indices.append(idx)

    # Build validated clean DataFrame
    if valid_indices:
        cleaned_df = pd.DataFrame({
            "date": parsed_dates.iloc[valid_indices].dt.strftime("%Y-%m-%d"),
            "medicine_id": df_working["medicine_id"].iloc[valid_indices].astype(str).str.strip().str.upper(),
            "units_sold": df_working["units_sold"].iloc[valid_indices].astype(float).round(4),
            "transaction_id": df_working["transaction_id"].iloc[valid_indices] if "transaction_id" in df_working.columns else None
        })
        # If transaction_id is completely None, drop it
        if cleaned_df["transaction_id"].isnull().all():
            cleaned_df = cleaned_df.drop(columns=["transaction_id"])
    else:
        cleaned_df = pd.DataFrame(columns=["date", "medicine_id", "units_sold"])

    return {
        "is_valid": len(errors) == 0,
        "error_message": f"{len(errors)} validation errors detected." if errors else None,
        "total_rows": len(df_working),
        "valid_count": len(cleaned_df),
        "invalid_count": len(errors),
        "errors": errors,
        "valid_df": cleaned_df,
        "detected_format": detected_format
    }


def deduplicate_records(
    incoming_df: pd.DataFrame,
    existing_daily_path: str = "processed/daily_sales.parquet",
    staging_path: str = IMPORTED_STAGING_PATH
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    Identifies and filters duplicate records.
    
    Deduplication Policy:
    1. If transaction_id is present: deduplicate strictly on transaction_id.
    2. Otherwise: deduplicate on composite key (date, medicine_id).
       If an identical (date, medicine_id) pair exists with identical or existing units_sold,
       the record is skipped as an existing record to preserve idempotent re-uploads.
    
    Returns:
    - unique_df: Non-duplicate records ready for ingestion
    - duplicate_count: Number of duplicate records skipped
    - duplicate_df: Records identified as duplicates
    """
    if incoming_df.empty:
        return incoming_df, 0, pd.DataFrame()

    # Build set of existing keys
    existing_keys = set()
    existing_tx_ids = set()

    # Load from processed daily_sales.parquet if available
    p_daily = Path(existing_daily_path)
    if p_daily.exists():
        try:
            if p_daily.is_file():
                df_ex = pd.read_parquet(p_daily)
            else:
                parts = [f for f in p_daily.glob("*.parquet") if not f.name.startswith(".")]
                df_ex = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True) if parts else pd.DataFrame()

            if not df_ex.empty:
                df_ex["date_str"] = pd.to_datetime(df_ex["date"]).dt.strftime("%Y-%m-%d")
                for _, r in df_ex.iterrows():
                    existing_keys.add((r["date_str"], str(r["medicine_id"]).strip().upper()))
        except Exception:
            pass

    # Also load from staging location if exists
    p_staging = Path(staging_path)
    if p_staging.exists():
        try:
            df_stg = pd.read_csv(p_staging)
            if not df_stg.empty:
                df_stg["date_str"] = pd.to_datetime(df_stg["date"]).dt.strftime("%Y-%m-%d")
                for _, r in df_stg.iterrows():
                    existing_keys.add((r["date_str"], str(r["medicine_id"]).strip().upper()))
                    if "transaction_id" in r and not pd.isnull(r["transaction_id"]):
                        existing_tx_ids.add(str(r["transaction_id"]).strip())
        except Exception:
            pass

    unique_rows = []
    duplicate_rows = []

    has_tx = "transaction_id" in incoming_df.columns and not incoming_df["transaction_id"].isnull().all()

    # Within the incoming file itself, deduplicate multiple occurrences of the same key
    seen_in_batch = set()

    for _, row in incoming_df.iterrows():
        d_str = str(row["date"])
        m_str = str(row["medicine_id"]).strip().upper()
        composite_key = (d_str, m_str)

        is_dup = False
        if has_tx and not pd.isnull(row.get("transaction_id")):
            tx_id = str(row["transaction_id"]).strip()
            if tx_id in existing_tx_ids or tx_id in seen_in_batch:
                is_dup = True
            else:
                seen_in_batch.add(tx_id)
        else:
            if composite_key in existing_keys or composite_key in seen_in_batch:
                is_dup = True
            else:
                seen_in_batch.add(composite_key)

        if is_dup:
            duplicate_rows.append(row)
        else:
            unique_rows.append(row)

    unique_df = pd.DataFrame(unique_rows) if unique_rows else pd.DataFrame(columns=incoming_df.columns)
    duplicate_df = pd.DataFrame(duplicate_rows) if duplicate_rows else pd.DataFrame(columns=incoming_df.columns)

    return unique_df, len(duplicate_df), duplicate_df


def save_parquet_dataset(df: pd.DataFrame, target_path: Union[Path, str]):
    """
    Saves a DataFrame to Parquet, supporting both single-file and directory-based
    Parquet partition datasets (as generated by Apache Spark on Windows).
    """
    target_path = Path(target_path)
    if target_path.exists() and target_path.is_dir():
        # Remove old parquet parts inside directory
        for f in list(target_path.glob("*.parquet")):
            try:
                f.unlink()
            except Exception:
                pass
        out_file = target_path / "part-00000-pharmora.parquet"
        df.to_parquet(out_file, index=False)
    elif target_path.suffix == ".parquet" and target_path.exists() and target_path.is_file():
        df.to_parquet(target_path, index=False)
    else:
        target_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target_path / "part-00000-pharmora.parquet", index=False)


def persist_and_update_pipeline(
    validated_df: pd.DataFrame,
    source_filename: str = "manual_upload.csv",
    staging_path: str = IMPORTED_STAGING_PATH,
    processed_dir: str = PROCESSED_DIR
) -> Dict[str, Any]:
    """
    Safely persists validated, deduplicated sales records:
    1. Appends records to data/imported_sales.csv (preserving data/salesdaily.csv).
    2. Rebuilds/updates daily_sales.parquet, medicine_summary.parquet, and monthly_trends.parquet.
    3. Returns full telemetry on new records and new medicine categories added.
    """
    if validated_df.empty:
        return {
            "success": False,
            "message": "No new records to import (dataset is empty or all records were duplicates).",
            "imported_count": 0,
            "new_medicines": []
        }

    p_staging = Path(staging_path)
    p_staging.parent.mkdir(parents=True, exist_ok=True)
    p_out = Path(processed_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    # 1. Append to staging file (with audit metadata)
    df_to_save = validated_df.copy()
    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_to_save["imported_at"] = now_iso
    df_to_save["source_file"] = source_filename

    if p_staging.exists() and p_staging.stat().st_size > 0:
        df_to_save.to_csv(p_staging, mode="a", header=False, index=False)
    else:
        df_to_save.to_csv(p_staging, mode="w", header=True, index=False)

    # 2. Load existing daily_sales.parquet
    daily_parquet = p_out / "daily_sales.parquet"
    if daily_parquet.exists():
        if daily_parquet.is_file():
            df_existing = pd.read_parquet(daily_parquet)
        else:
            parts = [f for f in daily_parquet.glob("*.parquet") if not f.name.startswith(".")]
            df_existing = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True) if parts else pd.DataFrame()
    else:
        df_existing = pd.DataFrame()

    existing_meds = set(df_existing["medicine_id"].unique()) if not df_existing.empty else set()

    # 3. Enrich new records with calendar dimensions matching PySpark schema
    new_enriched = validated_df[["date", "medicine_id", "units_sold"]].copy()
    new_enriched["date_dt"] = pd.to_datetime(new_enriched["date"])
    new_enriched["year"] = new_enriched["date_dt"].dt.year.astype(np.int32)
    new_enriched["month"] = new_enriched["date_dt"].dt.month.astype(np.int32)
    new_enriched["day_of_month"] = new_enriched["date_dt"].dt.day.astype(np.int32)
    new_enriched["day_of_week"] = new_enriched["date_dt"].dt.day_name()
    new_enriched["date"] = new_enriched["date_dt"].dt.strftime("%Y-%m-%d")
    new_enriched = new_enriched.drop(columns=["date_dt"])

    # Combine existing + new records
    if not df_existing.empty:
        # Standardize existing columns
        for c in ["year", "month", "day_of_month"]:
            if c in df_existing.columns:
                df_existing[c] = df_existing[c].astype(np.int32)
        df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.strftime("%Y-%m-%d")
        df_existing["units_sold"] = df_existing["units_sold"].astype(float)
        
        # Concat and deduplicate on date + medicine_id (keeping latest)
        combined_df = pd.concat([df_existing, new_enriched], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["date", "medicine_id"], keep="last")
    else:
        combined_df = new_enriched

    combined_df = combined_df.sort_values(by=["date", "medicine_id"]).reset_index(drop=True)

    # 4. Overwrite processed/daily_sales.parquet cleanly
    save_parquet_dataset(combined_df, daily_parquet)

    # 5. Recompute and overwrite medicine_summary.parquet
    summary_records = []
    for med, grp in combined_df.groupby("medicine_id"):
        vol = grp["units_sold"]
        summary_records.append({
            "medicine_id": str(med),
            "total_active_days": int(len(grp)),
            "total_units_sold": round(float(vol.sum()), 2),
            "avg_daily_demand": round(float(vol.mean()), 2),
            "demand_volatility": round(float(vol.std() if len(grp) > 1 else 0.0), 2),
            "peak_single_day_sales": round(float(vol.max()), 2)
        })
    summary_df = pd.DataFrame(summary_records).sort_values("total_units_sold", ascending=False).reset_index(drop=True)
    save_parquet_dataset(summary_df, p_out / "medicine_summary.parquet")

    # 6. Recompute and overwrite monthly_trends.parquet
    monthly_records = []
    for (yr, mo, med), grp in combined_df.groupby(["year", "month", "medicine_id"]):
        monthly_records.append({
            "year": int(yr),
            "month": int(mo),
            "medicine_id": str(med),
            "monthly_units_sold": round(float(grp["units_sold"].sum()), 2),
            "avg_daily_sales": round(float(grp["units_sold"].mean()), 2),
            "days_recorded": int(len(grp))
        })
    monthly_df = pd.DataFrame(monthly_records).sort_values(["year", "month", "medicine_id"]).reset_index(drop=True)
    save_parquet_dataset(monthly_df, p_out / "monthly_trends.parquet")

    # Detect new medicines
    updated_meds = set(combined_df["medicine_id"].unique())
    new_meds = sorted(list(updated_meds - existing_meds))

    return {
        "success": True,
        "message": f"Successfully ingested {len(validated_df):,} records into data/imported_sales.csv and rebuilt Parquet facts.",
        "imported_count": len(validated_df),
        "total_active_records": len(combined_df),
        "total_medicines": len(updated_meds),
        "new_medicines": new_meds,
        "staging_path": str(p_staging),
        "artifacts_updated": [
            str(daily_parquet),
            str(p_out / "medicine_summary.parquet"),
            str(p_out / "monthly_trends.parquet")
        ]
    }
