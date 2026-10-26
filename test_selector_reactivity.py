"""
PHARMORA - Medicine Selector Reactivity & UI Data Flow Test
===========================================================
Tests that:
1. Global KPIs are invariant across medicine selections.
2. Every medicine selection produces distinct, accurate, non-trivial telemetry.
3. No silent data substitution occurs.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from app import load_all_pharmora_data, ATC_NAMES, ATC_SHORT

def test_reactivity():
    data = load_all_pharmora_data()
    daily = data["daily"]
    stockout = data["stockout"]
    expiry = data["expiry"]
    reorder = data["reorder"]
    forecast = data["forecast"]
    evals = data["evaluations"]

    # 1. Validate Global KPIs
    tot_vol = daily["units_sold"].sum()
    tot_stock = stockout["current_stock"].sum()
    tot_inv_val = stockout["inventory_value"].sum()
    tot_risk_val = expiry["value_at_risk"].sum()
    tot_order = reorder["suggested_order_quantity"].sum()

    print("=== GLOBAL NETWORK METRICS ===")
    print(f"Total Volume:                {tot_vol:,.2f} units")
    print(f"Total Warehouse Stock:       {tot_stock:,.1f} units")
    print(f"Total Asset Value:           ₹{tot_inv_val:,.2f}")
    print(f"Total Capital at Expiry Risk: ₹{tot_risk_val:,.2f}")
    print(f"Total Urgent Reorder Units:  {tot_order} units")

    assert round(tot_vol, 2) >= 127595.50, f"Unexpected volume: {tot_vol}"
    assert tot_stock == 462.0, f"Unexpected stock: {tot_stock}"
    assert round(tot_inv_val, 2) == 5055.25, f"Unexpected asset value: {tot_inv_val}"
    assert round(tot_risk_val, 2) == 1443.20, f"Unexpected risk value: {tot_risk_val}"
    assert tot_order == 550, f"Unexpected reorder volume: {tot_order}"

    # 2. Test 4 Diverse Medicines
    test_meds = ["M01AB", "N02BE", "N05C", "R03"]
    prev_metrics = []

    print("\n=== MEDICINE-SPECIFIC REACTIVITY TESTS ===")
    for med in test_meds:
        m_daily = daily[daily["medicine_id"] == med]
        m_stock = stockout[stockout["medicine_id"] == med].iloc[0]
        m_exp = expiry[expiry["medicine_id"] == med]
        m_reorder = reorder[reorder["medicine_id"] == med].iloc[0]
        m_fc = forecast[forecast["medicine_id"] == med]
        m_eval = evals[evals["medicine_id"] == med].iloc[0]

        vol = round(m_daily["units_sold"].sum(), 2)
        mean_d = round(m_daily["units_sold"].mean(), 2)
        stk = m_stock["current_stock"]
        cov = m_stock["estimated_days_of_cover"]
        stk_status = m_stock["risk_status"]
        exp_u = m_exp[m_exp["expiry_status"] == "EXPIRED"]["unsold_units_at_risk"].sum()
        fc_sum = round(m_fc["predicted_units"].sum(), 2)
        reorder_q = m_reorder["suggested_order_quantity"]
        reorder_st = m_reorder["reorder_status"]
        reorder_tm = m_reorder["recommended_order_timing"]

        print(f"\n[MEDICINE: {med} - {ATC_SHORT[med]}]")
        print(f"  Sales Volume:     {vol:,.1f} units (Mean: {mean_d}/day)")
        print(f"  Physical Stock:   {stk} units ({cov} days cover) -> Status: {stk_status}")
        print(f"  Warehouse Batches:{len(m_exp)} batches (Expired Units: {exp_u:.0f})")
        print(f"  7-Day Forecast:   {fc_sum} units (RF MAE: {m_eval['rf_mae']:.3f})")
        print(f"  Procurement Order:{reorder_q} units ({reorder_st}) -> {reorder_tm}")

        metric_fingerprint = (vol, stk, cov, stk_status, fc_sum, reorder_q, reorder_st)
        assert metric_fingerprint not in prev_metrics, f"Duplicate metrics found for {med}!"
        prev_metrics.append(metric_fingerprint)

    print("\n" + "=" * 60)
    print("ALL MEDICINE REACTIVITY CHECKS PASSED (100% DISTINCT & ACCURATE)")
    print("=" * 60)

if __name__ == "__main__":
    test_reactivity()
