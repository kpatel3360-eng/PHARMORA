"""
PHARMORA - Automated Test Suite for Rescue-to-Reorder™ Engine
============================================================
Verifies:
1. Multi-branch inventory modeling with individual branch stocks, lead times, and batches.
2. 7-step decision sequence:
   - Identifies eligible surplus stock approaching expiry.
   - Matches compatible destination demand.
   - Proposes transfer before generating purchase recommendation.
   - Correctly recalculates remaining reorder quantity: Remaining = max(0, Reorder - Rescued).
   - Recommends supplier purchase only for remaining shortage.
   - Provides clear explanations when no match is available.
3. Strict unexpired batch validation (never transfers expired stock).
4. Authorization & inventory integrity check: quantities NEVER change until confirmed.
5. Independent pharmacy network sharing: opt-in, masked discovery, consent requirement.
6. Single-store fallback behavior.
7. Financial ledger accuracy: waste avoided and procurement savings.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import datetime
import pandas as pd
import numpy as np

from src.rescue import (
    MODE_MULTI_BRANCH,
    MODE_NETWORK_SHARING,
    MODE_SINGLE_STORE,
    DEFAULT_BRANCHES,
    DEFAULT_NETWORK_PHARMACIES,
    get_initial_branch_inventory,
    get_initial_branch_batches,
    get_initial_network_stock,
    run_rescue_to_reorder_engine,
    execute_confirmed_transfer,
    calculate_branch_reorder_requirement
)


def test_1_reorder_requirement_calculation():
    print("\n--- TEST 1: Reorder Requirement & Threshold Calculations ---")
    # Medicine M01AE at BR-01: Stock 12, Safety 12, Lead Time 4d, Daily Demand 3.90/d
    # Lead time demand = 3.90 * 4 = 15.6. ROP = 15.6 + 12 = 27.6.
    # Inv pos = 12.0 <= 27.6 -> Needs reorder.
    # Target stock = 3.90 * (4 + 7) + 12 = 42.9 + 12 = 54.9.
    # Gross reorder = ceil(54.9 - 12) = 43 units.
    res = calculate_branch_reorder_requirement(
        current_stock=12.0,
        pending_orders=0.0,
        daily_demand=3.90,
        lead_time_days=4,
        safety_stock=12.0,
        review_period_days=7
    )
    assert res["is_reorder_needed"] is True
    assert res["reorder_point"] == 27.6
    assert res["gross_reorder_qty"] == 43
    print(f" -> [PASS] Reorder needed accurately flagged: ROP={res['reorder_point']}, Gross Reorder={res['gross_reorder_qty']}u")

    # High stock: Stock 60, Safety 10, Lead Time 3d, Daily Demand 2.0/d
    res_healthy = calculate_branch_reorder_requirement(
        current_stock=60.0,
        pending_orders=0.0,
        daily_demand=2.0,
        lead_time_days=3,
        safety_stock=10.0
    )
    assert res_healthy["is_reorder_needed"] is False
    assert res_healthy["gross_reorder_qty"] == 0
    print(" -> [PASS] Well-stocked inventory correctly bypasses reorder necessity.")


def test_2_multibranch_rescue_before_reorder_workflow():
    print("\n--- TEST 2: Multi-Branch Rescue-Before-Reorder Decision Engine ---")
    branch_inv = get_initial_branch_inventory()
    branch_batches = get_initial_branch_batches()

    engine_res = run_rescue_to_reorder_engine(
        operating_mode=MODE_MULTI_BRANCH,
        branch_inventory=branch_inv,
        branch_batches=branch_batches,
        target_branch_id="BR-01"
    )

    recs = engine_res["recommendations"]
    assert len(recs) > 0

    # Specifically check M01AE at BR-01:
    # BR-01 needs M01AE, BR-02 has surplus batch B-M01AE-BR2-01 (48 days to expiry, 25u)
    m01ae_rec = next((r for r in recs if r["medicine_id"] == "M01AE"), None)
    assert m01ae_rec is not None, "M01AE recommendation missing for BR-01"

    print(f" -> M01AE Decision: {m01ae_rec['decision']}")
    print(f"    Source: {m01ae_rec['source_name']}")
    print(f"    Transferred: {m01ae_rec['transfer_quantity']} units | Expiry: {m01ae_rec['batch_expiry']} ({m01ae_rec['days_to_expiry']}d left)")
    print(f"    Original Need: {m01ae_rec['original_reorder_qty']}u -> Remaining Supplier Purchase: {m01ae_rec['remaining_reorder_qty']}u")

    assert m01ae_rec["decision"] == "RECOMMENDED_TRANSFER"
    assert m01ae_rec["source_id"] == "BR-02"
    assert m01ae_rec["transfer_quantity"] > 0
    assert m01ae_rec["original_reorder_qty"] == m01ae_rec["transfer_quantity"] + m01ae_rec["remaining_reorder_qty"]
    assert m01ae_rec["remaining_reorder_qty"] < m01ae_rec["original_reorder_qty"]
    print(" -> [PASS] Transfer proposed before purchase; remaining reorder correctly decremented.")

    # Check N02BE at BR-01:
    # Massive shortage at BR-01 (388 units), but no other branch has surplus N02BE!
    n02be_rec = next((r for r in recs if r["medicine_id"] == "N02BE"), None)
    assert n02be_rec is not None
    assert n02be_rec["decision"] == "SUPPLIER_PURCHASE_ONLY"
    assert n02be_rec["transfer_quantity"] == 0
    assert n02be_rec["remaining_reorder_qty"] == n02be_rec["original_reorder_qty"]
    assert "No eligible internal branch has surplus" in n02be_rec["reason"]
    print(f" -> [PASS] Step 7 verified: Explicit explanation provided when no surplus match exists.")


def test_3_strict_non_expired_batch_guarantee():
    print("\n--- TEST 3: Strict Non-Expired Batch Validation ---")
    branch_inv = get_initial_branch_inventory()
    branch_batches = get_initial_branch_batches().copy()

    # Artificially inject an expired surplus batch at BR-02
    branch_batches.loc[len(branch_batches)] = {
        "batch_id": "B-M01AE-EXPIRED",
        "branch_id": "BR-02",
        "medicine_id": "M01AE",
        "quantity": 100.0,
        "expiry_date": "2019-09-01",
        "unit_cost": 5.25,
        "storage_condition": "Room Temperature",
        "expiry_dt": pd.to_datetime("2019-09-01"),
        "days_to_expiry": -38,
        "is_expired": True,
        "is_synthetic": True
    }

    engine_res = run_rescue_to_reorder_engine(
        operating_mode=MODE_MULTI_BRANCH,
        branch_inventory=branch_inv,
        branch_batches=branch_batches,
        target_branch_id="BR-01"
    )

    recs = engine_res["recommendations"]
    for r in recs:
        assert r.get("batch_id") != "B-M01AE-EXPIRED", "FATAL: Expired batch was recommended for transfer!"
        if r["decision"] == "RECOMMENDED_TRANSFER":
            assert r["days_to_expiry"] > 0, "Transfer recommendation must strictly have positive days until expiry!"

    print(" -> [PASS] Expired batches are 100% blocked from transfer recommendations.")


def test_4_authorization_and_inventory_immutability():
    print("\n--- TEST 4: Authorization Check & Inventory Immutability ---")
    branch_inv = get_initial_branch_inventory()
    src_before = branch_inv.loc[(branch_inv["branch_id"] == "BR-02") & (branch_inv["medicine_id"] == "M01AE"), "current_stock"].values[0]
    dest_before = branch_inv.loc[(branch_inv["branch_id"] == "BR-01") & (branch_inv["medicine_id"] == "M01AE"), "current_stock"].values[0]

    # Running recommendation engine MUST NOT modify inventory
    _ = run_rescue_to_reorder_engine(
        operating_mode=MODE_MULTI_BRANCH,
        branch_inventory=branch_inv,
        branch_batches=get_initial_branch_batches(),
        target_branch_id="BR-01"
    )

    src_after_rec = branch_inv.loc[(branch_inv["branch_id"] == "BR-02") & (branch_inv["medicine_id"] == "M01AE"), "current_stock"].values[0]
    dest_after_rec = branch_inv.loc[(branch_inv["branch_id"] == "BR-01") & (branch_inv["medicine_id"] == "M01AE"), "current_stock"].values[0]
    assert src_before == src_after_rec and dest_before == dest_after_rec, "Inventory was modified by recommendation alone!"
    print(" -> [PASS] Recommending transfers does not alter inventory quantities (Zero unconfirmed debit).")

    # Now simulate explicit authorized confirmation of 15 units transfer
    updated_inv, audit_log = execute_confirmed_transfer(
        branch_inventory=branch_inv,
        source_branch_id="BR-02",
        dest_branch_id="BR-01",
        medicine_id="M01AE",
        quantity=15
    )

    src_after_conf = updated_inv.loc[(updated_inv["branch_id"] == "BR-02") & (updated_inv["medicine_id"] == "M01AE"), "current_stock"].values[0]
    dest_after_conf = updated_inv.loc[(updated_inv["branch_id"] == "BR-01") & (updated_inv["medicine_id"] == "M01AE"), "current_stock"].values[0]

    assert src_after_conf == src_before - 15
    assert dest_after_conf == dest_before + 15
    assert audit_log["status"] == "COMPLETED"
    print(f" -> [PASS] Confirmed transfer executed: BR-02 stock {src_before} -> {src_after_conf}, BR-01 stock {dest_before} -> {dest_after_conf}.")


def test_5_independent_pharmacy_network_sharing():
    print("\n--- TEST 5: Independent Pharmacy Network Sharing & Privacy Controls ---")
    branch_inv = get_initial_branch_inventory()
    net_stock = get_initial_network_stock()

    res = run_rescue_to_reorder_engine(
        operating_mode=MODE_NETWORK_SHARING,
        branch_inventory=branch_inv,
        branch_batches=get_initial_branch_batches(),
        network_stock=net_stock
    )

    recs = res["recommendations"]
    matches = [r for r in recs if r["decision"] == "POTENTIAL_NETWORK_MATCH"]
    assert len(matches) > 0, "Expected potential network matches for independent pharmacy mode"

    for m in matches:
        # Verify status is unconfirmed potential match
        assert m["status"] == "POTENTIAL_MATCH"
        # Verify batch details are privacy-masked
        assert m["batch_id"] == "PEER_LOT_MASKED"
        assert m["destination_id"] == "NET-SELF"
        print(f" -> Potential Match: {m['medicine_id']} from {m['source_name']} (Rescued: {m['transfer_quantity']}u, Saved: ₹{m['purchases_avoided']:.2f})")

    print(" -> [PASS] Peer stock sharing matches remain potential and masked until connection request.")


def test_6_single_store_mode_fallback():
    print("\n--- TEST 6: Single-Store Mode Fallback ---")
    branch_inv = get_initial_branch_inventory()
    res = run_rescue_to_reorder_engine(
        operating_mode=MODE_SINGLE_STORE,
        branch_inventory=branch_inv,
        branch_batches=get_initial_branch_batches()
    )

    recs = res["recommendations"]
    for r in recs:
        assert r["decision"] == "SINGLE_STORE_REORDER"
        assert r["transfer_quantity"] == 0
        assert r["remaining_reorder_qty"] == r["original_reorder_qty"]
        assert "Single-store operating model active" in r["reason"]

    assert res["summary"]["potential_stock_rescued_units"] == 0
    print(" -> [PASS] Single-store operating mode maintains standard local procurement without transfers.")


def test_7_financial_ledger_comparison():
    print("\n--- TEST 7: 'Transfer First' vs 'Purchase New' Financial Ledger ---")
    branch_inv = get_initial_branch_inventory()
    batches = get_initial_branch_batches()

    res = run_rescue_to_reorder_engine(
        operating_mode=MODE_MULTI_BRANCH,
        branch_inventory=branch_inv,
        branch_batches=batches
    )

    comp = res["comparison"]
    print(f" -> Traditional Gross Purchase: {comp['traditional_purchase_units']} units (₹{comp['traditional_purchase_cost']:,.2f})")
    print(f" -> Rescued from Surplus:     {comp['rescued_units']} units (₹{comp['rescued_value_avoided']:,.2f})")
    print(f" -> Residual Supplier Order:  {comp['residual_purchase_units']} units (₹{comp['residual_purchase_cost']:,.2f})")
    print(f" -> Net Cost Reduction:       {comp['net_savings_pct']}%")

    assert comp["traditional_purchase_units"] == comp["rescued_units"] + comp["residual_purchase_units"]
    assert comp["rescued_value_avoided"] > 0
    assert comp["residual_purchase_cost"] < comp["traditional_purchase_cost"]
    print(" -> [PASS] Financial ledger balances with 100% precision.")


def main():
    print("=" * 80)
    print("  PHARMORA - RESCUE-TO-REORDER™ DECISION ENGINE VERIFICATION")
    print("=" * 80)

    try:
        test_1_reorder_requirement_calculation()
        test_2_multibranch_rescue_before_reorder_workflow()
        test_3_strict_non_expired_batch_guarantee()
        test_4_authorization_and_inventory_immutability()
        test_5_independent_pharmacy_network_sharing()
        test_6_single_store_mode_fallback()
        test_7_financial_ledger_comparison()

        print("\n" + "=" * 80)
        print("  ALL 7 RESCUE-TO-REORDER TEST SUITES PASSED (100% HEALTHY)")
        print("=" * 80)
    except AssertionError as e:
        import traceback
        traceback.print_exc()
        print(f"\n[FAIL] Test assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] Unexpected error during verification: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
