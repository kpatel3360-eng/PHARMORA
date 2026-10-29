"""
PHARMORA - Smart Reordering Engine (Phase 4: Automated Replenishment Intelligence)
==================================================================================
Generates data-driven procurement recommendations answering:
1. What medicine to reorder (medicine_id)
2. How much to order (suggested_order_quantity)
3. When to order it (recommended_order_timing & reorder_status)

Technical Core:
- Target Stock (Order-Up-To Level): Forecast demand over Lead Time (L) + Review Period (R) + Safety Stock.
- Inventory Position: Current Stock + Pending Order Quantity (prevents duplicate orders).
- Reorder Point (ROP): Lead-Time Demand + Statistical Safety Stock.
- Timing Horizons: URGENT (Immediate), UPCOMING (within review period), NOT_REQUIRED.
- Healthcare Compliance: Integrates batch expiry risk audit from Phase 3, flags near-expiry batches
  for supervisor review, and strictly prohibits selling expired medicines.
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


VALID_MEDICINE_IDS = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
DEFAULT_REVIEW_PERIOD_DAYS = 7


def _read_data_file(path: Path) -> pd.DataFrame:
    """Reads a CSV or Parquet file or Parquet directory partition robustly."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found at '{path.resolve()}'.")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.is_file():
        return pd.read_parquet(path)
    parts = [f for f in path.glob("*.parquet") if not f.name.startswith(".")]
    if not parts:
        raise FileNotFoundError(f"No valid Parquet files found in directory '{path}'.")
    return pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)


def calculate_reorder_recommendations(
    inventory_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    expiry_df: Optional[pd.DataFrame] = None,
    sales_df: Optional[pd.DataFrame] = None,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    min_order_quantities: Optional[Dict[str, int]] = None,
    allowed_medicines: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Computes optimal replenishment recommendations across all medicines:
    - Inventory Position = Current Stock + Pending Orders
    - Lead-Time Demand = ML forecast demand over supplier lead time
    - Reorder Point (ROP) = Lead-Time Demand + Safety Stock
    - Target Stock (S) = Forecast demand over (Lead Time + Review Period) + Safety Stock
    - Suggested Order Quantity = ceil(max(0, Target Stock - Inventory Position)) when Inv Position <= ROP, else 0
    - Incorporates pending orders, supplier minimum order quantities (MOQ), and expiry risk audits.
    """
    inv = inventory_df.copy()
    fc = forecast_df.copy()
    moq_dict = min_order_quantities or {}

    # Validation: schema checks
    required_inv_cols = [
        "medicine_id", "current_stock", "supplier_lead_time_days",
        "safety_stock", "pending_order_quantity"
    ]
    missing = [c for c in required_inv_cols if c not in inv.columns]
    if missing:
        raise ValueError(f"Missing required columns in inventory dataset: {missing}")

    # Coerce numeric values and filter invalid records
    inv["medicine_id"] = inv["medicine_id"].astype(str)
    inv["current_stock"] = pd.to_numeric(inv["current_stock"], errors="coerce").fillna(0.0).clip(lower=0.0)
    inv["pending_order_quantity"] = pd.to_numeric(inv["pending_order_quantity"], errors="coerce").fillna(0.0).clip(lower=0.0)
    inv["supplier_lead_time_days"] = pd.to_numeric(inv["supplier_lead_time_days"], errors="coerce").fillna(7).astype(int).clip(lower=1)
    inv["safety_stock"] = pd.to_numeric(inv["safety_stock"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Filter to known target medicines and ensure unique records
    valid_ids = allowed_medicines if allowed_medicines is not None else VALID_MEDICINE_IDS
    inv = inv[inv["medicine_id"].isin(valid_ids)].copy()
    if inv["medicine_id"].duplicated().any():
        raise ValueError(f"Duplicate medicine IDs detected in inventory dataset: {inv[inv['medicine_id'].duplicated()]['medicine_id'].tolist()}")

    # Prepare daily forecast lookup
    fc["forecast_date"] = pd.to_datetime(fc["forecast_date"])
    forecast_lookup = {}
    for med, group in fc.groupby("medicine_id"):
        forecast_lookup[med] = group.sort_values("forecast_date")["predicted_units"].tolist()

    # Fallback historical average daily demand
    hist_avg = {}
    if sales_df is not None and not sales_df.empty:
        hist_avg = sales_df.groupby("medicine_id")["units_sold"].mean().to_dict()

    # Prepare expiry risk summary if available
    expiry_summary = {}
    if expiry_df is not None and not expiry_df.empty:
        exp_clean = expiry_df.copy()
        for med, group in exp_clean.groupby("medicine_id"):
            expired_units = float(group[group["expiry_status"] == "EXPIRED"]["unsold_units_at_risk"].sum())
            near_risk_units = float(group[group["expiry_status"] == "EXPIRY_RISK"]["unsold_units_at_risk"].sum())
            expiry_summary[med] = {
                "expired_units": round(expired_units, 2),
                "near_risk_units": round(near_risk_units, 2),
                "has_expired": expired_units > 0.0,
                "has_near_risk": near_risk_units > 0.0
            }

    recommendations = []

    for _, row in inv.iterrows():
        med = row["medicine_id"]
        stock = float(row["current_stock"])
        pending = float(row["pending_order_quantity"])
        lead_time = int(row["supplier_lead_time_days"])
        ss = float(row["safety_stock"])
        moq = moq_dict.get(med, 1)

        # 1. Calculate Inventory Position
        inv_position = stock + pending

        # 2. Daily forecast rates and Lead-Time Demand
        daily_fc = forecast_lookup.get(med, [])
        fc_len = len(daily_fc)
        daily_rate = float(np.mean(daily_fc)) if daily_fc else float(hist_avg.get(med, 5.0))

        if daily_fc:
            if lead_time <= fc_len:
                lt_demand = float(sum(daily_fc[:lead_time]))
            else:
                lt_demand = float(sum(daily_fc) + (lead_time - fc_len) * daily_rate)
        else:
            lt_demand = float(lead_time * daily_rate)

        # 3. Reorder Point (ROP)
        rop = lt_demand + ss

        # 4. Target Stock over (Lead Time + Review Period)
        target_horizon = lead_time + review_period_days
        if daily_fc:
            if target_horizon <= fc_len:
                target_demand = float(sum(daily_fc[:target_horizon]))
            else:
                target_demand = float(sum(daily_fc) + (target_horizon - fc_len) * daily_rate)
        else:
            target_demand = float(target_horizon * daily_rate)

        target_stock = target_demand + ss

        # 5. Days until Reorder Point is reached (Zero-Division Safe)
        if daily_rate > 0.0:
            days_until_rop = max(0.0, (inv_position - rop) / daily_rate)
        else:
            days_until_rop = 999.0 if inv_position > rop else 0.0

        # 6. Expiry Audit Context
        exp_info = expiry_summary.get(med, {
            "expired_units": 0.0,
            "near_risk_units": 0.0,
            "has_expired": False,
            "has_near_risk": False
        })

        # 7. Order Quantity & Status Determination
        reasons = []

        if inv_position <= rop:
            reorder_status = "URGENT"
            # Suggested order quantity brings inventory position up to Target Stock
            raw_order = max(0.0, target_stock - inv_position)
            # Ceiling to whole discrete packages/units
            suggested_qty = math.ceil(raw_order)

            # Apply supplier Minimum Order Quantity (MOQ) if specified
            if suggested_qty > 0 and suggested_qty < moq:
                suggested_qty = moq
                reasons.append(f"Order quantity adjusted up to supplier MOQ of {moq} units.")

            # Distinguish critical urgency (stockout before delivery) vs standard ROP breach
            if inv_position < lt_demand:
                recommended_timing = "IMMEDIATE (Within 24 Hours - Stockout Imminent)"
                reasons.append(
                    f"CRITICAL: Inventory position ({inv_position:.1f} units) cannot cover lead-time demand "
                    f"({lt_demand:.1f} units). Stockout guaranteed before replenishment arrives."
                )
            else:
                recommended_timing = "IMMEDIATE (Current Order Cycle)"
                reasons.append(
                    f"REORDER THRESHOLD BREACHED: Inventory position ({inv_position:.1f} units) is at or below "
                    f"ROP ({rop:.1f} units). Safety stock buffer ({ss:.1f} units) is being drawn down."
                )

            if pending > 0:
                reasons.append(f"In-flight pending order of {pending:.0f} units accounted for to prevent duplicate ordering.")

        elif days_until_rop <= review_period_days:
            reorder_status = "UPCOMING"
            suggested_qty = 0  # No order placed today
            recommended_timing = f"UPCOMING (In ~{days_until_rop:.1f} Days)"
            reasons.append(
                f"Sufficient for immediate needs, but inventory position ({inv_position:.1f} units) will deplete "
                f"to ROP ({rop:.1f} units) in approximately {days_until_rop:.1f} days (within {review_period_days}-day review window)."
            )
            if pending > 0:
                reasons.append(f"Pending order of {pending:.0f} units already scheduled.")

        else:
            reorder_status = "NOT_REQUIRED"
            suggested_qty = 0
            recommended_timing = "NOT_REQUIRED"
            reasons.append(
                f"Optimal inventory buffer ({inv_position:.1f} units). "
                f"Covers demand for ~{days_until_rop:.1f} days beyond ROP."
            )

        # 8. Append Healthcare Compliance & Expiry Risk Warnings
        if exp_info["has_expired"]:
            reasons.append(
                f"EXPIRED STOCK DETECTED: {exp_info['expired_units']:.1f} expired units found in warehouse. "
                "Must be quarantined immediately for disposal; NEVER sell expired medicines."
            )
        if exp_info["has_near_risk"]:
            reasons.append(
                f"EXPIRY RISK REVIEW: {exp_info['near_risk_units']:.1f} units across active batches are at risk "
                "of expiring unsold before consumption. Verify batch FEFO status before placing large replenishment orders."
            )

        final_reason = " ".join(reasons)

        recommendations.append({
            "medicine_id": med,
            "current_stock": round(stock, 2),
            "pending_order_quantity": round(pending, 2),
            "forecast_lead_time_demand": round(lt_demand, 2),
            "safety_stock": round(ss, 2),
            "reorder_point": round(rop, 2),
            "target_stock": round(target_stock, 2),
            "suggested_order_quantity": int(suggested_qty),
            "reorder_status": reorder_status,
            "recommended_order_timing": recommended_timing,
            "recommendation_reason": final_reason
        })

    return pd.DataFrame(recommendations)


def run_reordering_pipeline(
    inventory_path: str = "data/inventory.csv",
    forecast_path: str = "processed/forecast_7days.parquet",
    expiry_path: str = "processed/batch_expiry_risk.parquet",
    sales_path: str = "processed/daily_sales.parquet",
    output_dir: str = "processed",
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS
) -> Dict[str, Any]:
    """
    Executes the complete Phase 4 Smart Reordering Pipeline:
    1. Ingests warehouse inventory, machine-learning demand forecasts, and batch expiry audits.
    2. Computes lead-time demand, ROP, target stock, and integer order quantities.
    3. Categorizes urgency (URGENT, UPCOMING, NOT_REQUIRED) with estimated days until reorder.
    4. Validates outputs and saves results to Parquet and CSV in processed/.
    5. Prints formatted executive summary to terminal.
    """
    inv_file = Path(inventory_path)
    fc_file = Path(forecast_path)
    exp_file = Path(expiry_path)
    sales_file = Path(sales_path)
    out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("  PHARMORA - PHASE 4: SMART REORDERING ENGINE (PRESCRIPTIVE PROCUREMENT)")
    print("=" * 85)
    print(f" -> Review Period: {review_period_days} days")

    inv_df = _read_data_file(inv_file)
    fc_df = _read_data_file(fc_file)
    exp_df = _read_data_file(exp_file) if exp_file.exists() else None
    sales_df = _read_data_file(sales_file) if sales_file.exists() else None

    print(f" -> Ingested Inventory Records: {len(inv_df)} medicines from {inv_file.as_posix()}")
    print(f" -> Ingested Forecast Records:  {len(fc_df)} entries from {fc_file.as_posix()}")
    if exp_df is not None:
        print(f" -> Ingested Expiry Audits:     {len(exp_df)} batch audits from {exp_file.as_posix()}")

    # Run Reorder Optimization
    recommendations_df = calculate_reorder_recommendations(
        inventory_df=inv_df,
        forecast_df=fc_df,
        expiry_df=exp_df,
        sales_df=sales_df,
        review_period_days=review_period_days
    )

    # -----------------------------------------------------------------------
    # Formatted Terminal Summary
    # -----------------------------------------------------------------------
    print("\n" + "-" * 85)
    print(" [1/2] SMART REORDER RECOMMENDATIONS SUMMARY")
    print("-" * 85)

    display_cols = [
        "medicine_id", "current_stock", "pending_order_quantity", "forecast_lead_time_demand",
        "reorder_point", "target_stock", "suggested_order_quantity", "reorder_status", "recommended_order_timing"
    ]
    display_df = recommendations_df[display_cols].copy()
    display_df.columns = [
        "Medicine", "Stock", "Pending", "LT Demand",
        "ROP", "Target", "Order Qty", "Status", "Timing"
    ]
    print(display_df.to_string(index=False))

    urgent_count = (recommendations_df["reorder_status"] == "URGENT").sum()
    upcoming_count = (recommendations_df["reorder_status"] == "UPCOMING").sum()
    not_req_count = (recommendations_df["reorder_status"] == "NOT_REQUIRED").sum()
    total_ordered_units = recommendations_df["suggested_order_quantity"].sum()

    print(f"\n -> Procurement Status Summary: {urgent_count} URGENT | {upcoming_count} UPCOMING | {not_req_count} NOT REQUIRED")
    print(f" -> Total Recommended Reorder Volume: {total_ordered_units} units across {urgent_count} urgent medicines")

    # -----------------------------------------------------------------------
    # Detailed Reasoning & Healthcare Actions
    # -----------------------------------------------------------------------
    print("\n" + "-" * 85)
    print(" [2/2] DETAILED CLINICAL & OPERATIONAL RECOMMENDATION REASONS")
    print("-" * 85)
    for _, r in recommendations_df.iterrows():
        print(f"• [{r['medicine_id']}] Status: {r['reorder_status']} | Order: {r['suggested_order_quantity']} units | Timing: {r['recommended_order_timing']}")
        print(f"  Reason: {r['recommendation_reason']}\n")

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------
    reorder_parquet = out_dir / "reorder_recommendations.parquet"
    reorder_csv = out_dir / "reorder_recommendations.csv"

    recommendations_df.to_parquet(reorder_parquet, index=False)
    recommendations_df.to_csv(reorder_csv, index=False)

    print("=" * 85)
    print("  REORDERING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  -> Recommendations Table Saved: {reorder_parquet.as_posix()} ({len(recommendations_df)} rows)")
    print(f"  -> CSV Export Saved:            {reorder_csv.as_posix()} ({len(recommendations_df)} rows)")
    print("=" * 85)

    return {
        "recommendations": recommendations_df,
        "parquet_path": str(reorder_parquet),
        "csv_path": str(reorder_csv)
    }


def main():
    parser = argparse.ArgumentParser(description="PHARMORA Smart Reordering Engine")
    parser.add_argument(
        "--inventory",
        type=str,
        default="data/inventory.csv",
        help="Path to inventory master CSV (default: data/inventory.csv)"
    )
    parser.add_argument(
        "--forecast",
        type=str,
        default="processed/forecast_7days.parquet",
        help="Path to 7-day forecast table (default: processed/forecast_7days.parquet)"
    )
    parser.add_argument(
        "--expiry",
        type=str,
        default="processed/batch_expiry_risk.parquet",
        help="Path to batch expiry risk table (default: processed/batch_expiry_risk.parquet)"
    )
    parser.add_argument(
        "--sales",
        type=str,
        default="processed/daily_sales.parquet",
        help="Path to daily sales Parquet table (default: processed/daily_sales.parquet)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed",
        help="Path to output directory for recommendations (default: processed)"
    )
    parser.add_argument(
        "--review_period",
        type=int,
        default=DEFAULT_REVIEW_PERIOD_DAYS,
        help=f"Replenishment review period in days (default: {DEFAULT_REVIEW_PERIOD_DAYS})"
    )
    args = parser.parse_args()

    run_reordering_pipeline(
        inventory_path=args.inventory,
        forecast_path=args.forecast,
        expiry_path=args.expiry,
        sales_path=args.sales,
        output_dir=args.output_dir,
        review_period_days=args.review_period
    )


if __name__ == "__main__":
    main()
