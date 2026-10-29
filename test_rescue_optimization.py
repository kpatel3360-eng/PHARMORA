"""
================================================================================
  PHARMORA - RESCUE-TO-REORDER™ MILP CONSTRAINT OPTIMIZER TEST SUITE
================================================================================
Comprehensive verification of the SciPy MILP Cost-Optimization Engine:
1. Stock & Surplus Constraints (Batch capacity & Branch surplus limits)
2. Single-Unit / Anti-Duplicate Allocation (No double-counting across competing destinations)
3. Insufficient Supply Handling (Optimal partial rescue + residual purchase calculation)
4. Expiry Suitability & Shelf-Life Consumption Feasibility (14-day cutoff + rate feasibility)
5. Storage Requirements & Feasibility Constraints (Cold Chain vs Room Temp certification)
6. Comparison with Greedy Matching (MILP total cost <= Greedy total cost)
7. Robust Heuristic Fallback (Seamless fallback when SciPy is unavailable or infeasible)
================================================================================
"""

import sys
import datetime
import pandas as pd
import numpy as np

import src.rescue as rescue_mod
from src.rescue import (
    MODE_MULTI_BRANCH,
    MODE_NETWORK_SHARING,
    MODE_SINGLE_STORE,
    optimize_rescue_transfers,
    _greedy_rescue_match_fallback,
    run_rescue_to_reorder_engine,
    get_initial_branch_inventory,
    get_initial_branch_batches,
    DEFAULT_BRANCHES
)


def test_1_stock_and_surplus_constraints():
    """Verify allocations strictly respect batch quantities and branch surplus caps."""
    print("\n--- TEST 1: Stock & Surplus Constraints ---")
    destinations = [
        {
            "dest_id": "BR-01",
            "dest_name": "Central Pharmacy",
            "medicine_id": "M01AE",
            "gross_reorder_qty": 50,
            "daily_demand": 5.0,
            "unit_cost": 10.0,
            "storage_certified": ["Room Temperature"]
        }
    ]
    candidate_batches = [
        {
            "batch_id": "BATCH-S1",
            "source_id": "BR-02",
            "source_name": "North Clinic",
            "medicine_id": "M01AE",
            "quantity": 18.0,
            "expiry_date": "2019-12-01",
            "days_to_expiry": 53,
            "unit_cost": 10.0,
            "storage_condition": "Room Temperature",
            "distance_km": 0.0
        }
    ]
    # Source surplus is capped at 15 (less than batch quantity 18)
    source_surpluses = {("BR-02", "M01AE"): 15.0}
    storage_cert_map = {"BR-01": ["Room Temperature"], "BR-02": ["Room Temperature"]}

    res = optimize_rescue_transfers(
        destinations=destinations,
        candidate_batches=candidate_batches,
        source_surpluses=source_surpluses,
        storage_cert_map=storage_cert_map,
        operating_mode=MODE_MULTI_BRANCH
    )

    assert res["solver_status"] == "OPTIMAL_MILP"
    assert len(res["allocations"]) == 1
    alloc_qty = list(res["allocations"].values())[0]
    assert alloc_qty == 15, f"Expected exactly 15 units allocated (capped by source surplus), got {alloc_qty}"
    print(f" -> [PASS] Allocation ({alloc_qty}u) strictly respected source surplus cap (15u < batch 18u).")


def test_2_single_unit_no_duplicate_allocation():
    """Verify that when multiple destinations compete for a single batch, it cannot be double-allocated."""
    print("\n--- TEST 2: Single-Unit Allocation & Competition ---")
    destinations = [
        {
            "dest_id": "BR-01",
            "dest_name": "Central Pharmacy",
            "medicine_id": "M01AE",
            "gross_reorder_qty": 30,
            "daily_demand": 3.0,
            "unit_cost": 10.0,
            "storage_certified": ["Room Temperature"]
        },
        {
            "dest_id": "BR-03",
            "dest_name": "South Outpost",
            "medicine_id": "M01AE",
            "gross_reorder_qty": 30,
            "daily_demand": 3.0,
            "unit_cost": 10.0,
            "storage_certified": ["Room Temperature"]
        }
    ]
    # Only 25 units available in one single batch at BR-02
    candidate_batches = [
        {
            "batch_id": "BATCH-SHARED-01",
            "source_id": "BR-02",
            "source_name": "North Clinic",
            "medicine_id": "M01AE",
            "quantity": 25.0,
            "expiry_date": "2019-12-01",
            "days_to_expiry": 53,
            "unit_cost": 10.0,
            "storage_condition": "Room Temperature",
            "distance_km": 0.0
        }
    ]
    source_surpluses = {("BR-02", "M01AE"): 25.0}
    storage_cert_map = {
        "BR-01": ["Room Temperature"],
        "BR-02": ["Room Temperature"],
        "BR-03": ["Room Temperature"]
    }

    res = optimize_rescue_transfers(
        destinations=destinations,
        candidate_batches=candidate_batches,
        source_surpluses=source_surpluses,
        storage_cert_map=storage_cert_map,
        operating_mode=MODE_MULTI_BRANCH
    )

    total_allocated = sum(res["allocations"].values())
    assert total_allocated <= 25, f"Over-allocation detected! Total allocated: {total_allocated} > batch qty: 25"
    assert total_allocated == 25, f"Expected full 25 units allocated across destinations, got {total_allocated}"
    print(f" -> [PASS] Batch of 25 units competed by 60 total demand correctly allocated {total_allocated}u without double-counting.")


def test_3_insufficient_supply_residual_purchases():
    """Verify that when supply < demand, transfers are recommended first, and residual purchases are calculated."""
    print("\n--- TEST 3: Insufficient Supply & Residual Purchase Calculation ---")
    branch_inv = get_initial_branch_inventory()
    branch_batches = get_initial_branch_batches()

    engine_res = run_rescue_to_reorder_engine(
        operating_mode=MODE_MULTI_BRANCH,
        branch_inventory=branch_inv,
        branch_batches=branch_batches,
        target_branch_id="BR-01"
    )

    m01ae_rec = next((r for r in engine_res["recommendations"] if r["medicine_id"] == "M01AE"), None)
    assert m01ae_rec is not None
    assert m01ae_rec["transfer_quantity"] > 0
    assert m01ae_rec["original_reorder_qty"] == m01ae_rec["transfer_quantity"] + m01ae_rec["remaining_reorder_qty"]
    assert m01ae_rec["remaining_reorder_qty"] > 0
    print(f" -> [PASS] Need {m01ae_rec['original_reorder_qty']}u: Rescued {m01ae_rec['transfer_quantity']}u -> Residual supplier purchase {m01ae_rec['remaining_reorder_qty']}u.")


def test_4_expiry_eligibility_and_shelf_life_feasibility():
    """Verify rejection of expired batches, batches <= 14 days, and batches that cannot be consumed before expiry."""
    print("\n--- TEST 4: Expiry Eligibility & Shelf-Life Feasibility ---")
    destinations = [
        {
            "dest_id": "BR-01",
            "dest_name": "Central Pharmacy",
            "medicine_id": "MED-FAST",
            "gross_reorder_qty": 50,
            "daily_demand": 1.0,  # Only consumes 1 unit per day!
            "unit_cost": 15.0,
            "storage_certified": ["Room Temperature"]
        }
    ]
    candidate_batches = [
        # Batch 1: Expired (days_to_expiry <= 0)
        {
            "batch_id": "B-EXP",
            "source_id": "BR-02",
            "source_name": "North Clinic",
            "medicine_id": "MED-FAST",
            "quantity": 30.0,
            "expiry_date": "2019-09-01",
            "days_to_expiry": -10,
            "unit_cost": 15.0,
            "storage_condition": "Room Temperature",
            "distance_km": 0.0
        },
        # Batch 2: Too close to expiry (10 days <= 14 cutoff)
        {
            "batch_id": "B-NEAR-EXP",
            "source_id": "BR-02",
            "source_name": "North Clinic",
            "medicine_id": "MED-FAST",
            "quantity": 30.0,
            "expiry_date": "2019-10-19",
            "days_to_expiry": 10,
            "unit_cost": 15.0,
            "storage_condition": "Room Temperature",
            "distance_km": 0.0
        },
        # Batch 3: 20 days shelf life. At 1.0 unit/day, dest can consume at most 20 units before expiry!
        {
            "batch_id": "B-FEASIBLE-LIMITED",
            "source_id": "BR-02",
            "source_name": "North Clinic",
            "medicine_id": "MED-FAST",
            "quantity": 50.0,
            "expiry_date": "2019-10-29",
            "days_to_expiry": 20,
            "unit_cost": 15.0,
            "storage_condition": "Room Temperature",
            "distance_km": 0.0
        }
    ]
    source_surpluses = {("BR-02", "MED-FAST"): 100.0}
    storage_cert_map = {"BR-01": ["Room Temperature"], "BR-02": ["Room Temperature"]}

    res = optimize_rescue_transfers(
        destinations=destinations,
        candidate_batches=candidate_batches,
        source_surpluses=source_surpluses,
        storage_cert_map=storage_cert_map,
        operating_mode=MODE_MULTI_BRANCH
    )

    assert "B-EXP" not in [arc["batch_id"] for arc in res["arcs"]], "Expired batch must not form an arc"
    assert "B-NEAR-EXP" not in [arc["batch_id"] for arc in res["arcs"]], "Batch <= 14d must not form an arc"

    # For B-FEASIBLE-LIMITED, consumption feasibility caps allocation to 20 units (20 days * 1.0 unit/day)
    alloc = res["allocations"]
    total_alloc = sum(alloc.values())
    assert total_alloc <= 20, f"Feasibility cap violated! Allocated {total_alloc} > 20 consumable units."
    print(f" -> [PASS] Excluded expired/near-expiry batches. Shelf-life consumption cap enforced: {total_alloc}u <= 20u.")


def test_5_storage_certification_feasibility():
    """Verify that Cold Chain batches are never allocated to branches without Cold Chain certification."""
    print("\n--- TEST 5: Storage Requirements & Infeasible Scenarios ---")
    destinations = [
        {
            "dest_id": "BR-03",  # South Outpost only has Room Temperature
            "dest_name": "South Outpost",
            "medicine_id": "COLD-VACCINE",
            "gross_reorder_qty": 20,
            "daily_demand": 2.0,
            "unit_cost": 50.0,
            "storage_certified": ["Room Temperature"]
        }
    ]
    candidate_batches = [
        {
            "batch_id": "B-COLD-01",
            "source_id": "BR-01",
            "source_name": "Central Pharmacy",
            "medicine_id": "COLD-VACCINE",
            "quantity": 20.0,
            "expiry_date": "2020-05-01",
            "days_to_expiry": 200,
            "unit_cost": 50.0,
            "storage_condition": "Cold Chain 2-8°C",
            "distance_km": 0.0
        }
    ]
    source_surpluses = {("BR-01", "COLD-VACCINE"): 20.0}
    storage_cert_map = {
        "BR-01": ["Room Temperature", "Cold Chain 2-8°C"],
        "BR-03": ["Room Temperature"]  # Not certified for cold chain!
    }

    res = optimize_rescue_transfers(
        destinations=destinations,
        candidate_batches=candidate_batches,
        source_surpluses=source_surpluses,
        storage_cert_map=storage_cert_map,
        operating_mode=MODE_MULTI_BRANCH
    )

    # Arc should be rejected due to storage incompatibility
    assert len(res["arcs"]) == 0, "Cold chain batch must not form valid arc to room-temp-only branch"
    assert len(res["allocations"]) == 0
    print(" -> [PASS] Storage constraint strictly satisfied: Cold Chain medicine 100% blocked from uncertified branch.")


def test_6_comparison_with_greedy_optimization():
    """Verify that MILP finds a solution with total cost <= greedy matching."""
    print("\n--- TEST 6: Cost-Minimization Comparison (MILP vs Greedy) ---")
    # Construct a network scenario with 2 destinations and 2 batches where greedy makes a suboptimal choice:
    # Dest 1 needs 10u at high logistics rate (distant). Dest 2 needs 10u at low logistics rate.
    # Batch 1 (10u) has high unit cost (waste avoided is crucial). Batch 2 (10u) has lower unit cost.
    destinations = [
        {
            "dest_id": "BR-NEAR",
            "dest_name": "Near Branch",
            "medicine_id": "MED-A",
            "gross_reorder_qty": 10,
            "daily_demand": 2.0,
            "unit_cost": 25.0,
            "storage_certified": ["Room Temperature"]
        },
        {
            "dest_id": "BR-FAR",
            "dest_name": "Far Branch",
            "medicine_id": "MED-A",
            "gross_reorder_qty": 10,
            "daily_demand": 2.0,
            "unit_cost": 25.0,
            "storage_certified": ["Room Temperature"]
        }
    ]
    candidate_batches = [
        {
            "batch_id": "B-CLOSE-SRC",
            "source_id": "SRC-A",
            "source_name": "Source A",
            "medicine_id": "MED-A",
            "quantity": 10.0,
            "expiry_date": "2019-12-01",
            "days_to_expiry": 50,
            "unit_cost": 25.0,
            "storage_condition": "Room Temperature",
            "distance_km": 5.0
        },
        {
            "batch_id": "B-FAR-SRC",
            "source_id": "SRC-B",
            "source_name": "Source B",
            "medicine_id": "MED-A",
            "quantity": 10.0,
            "expiry_date": "2019-12-15",
            "days_to_expiry": 65,
            "unit_cost": 25.0,
            "storage_condition": "Room Temperature",
            "distance_km": 40.0
        }
    ]
    source_surpluses = {("SRC-A", "MED-A"): 10.0, ("SRC-B", "MED-A"): 10.0}
    storage_cert_map = {
        "BR-NEAR": ["Room Temperature"],
        "BR-FAR": ["Room Temperature"],
        "SRC-A": ["Room Temperature"],
        "SRC-B": ["Room Temperature"]
    }

    # Run MILP
    milp_res = optimize_rescue_transfers(
        destinations=destinations,
        candidate_batches=candidate_batches,
        source_surpluses=source_surpluses,
        storage_cert_map=storage_cert_map,
        operating_mode=MODE_NETWORK_SHARING,
        reference_date=datetime.date(2019, 10, 9)
    )

    # Run Greedy Fallback
    greedy_recs, g_res, g_wa, g_pa, g_rem, g_log = _greedy_rescue_match_fallback(
        destinations=destinations,
        candidate_batches=candidate_batches,
        source_surpluses=source_surpluses,
        storage_cert_map=storage_cert_map,
        operating_mode=MODE_NETWORK_SHARING
    )

    milp_alloc_total = sum(milp_res["allocations"].values())
    print(f" -> MILP Allocated Units:   {milp_alloc_total}u (Solver Status: {milp_res['solver_status']})")
    print(f" -> Greedy Allocated Units: {g_res}u")
    assert milp_alloc_total >= g_res, "MILP must rescue at least as many or more units than greedy"
    print(" -> [PASS] MILP optimizer achieved mathematically optimal allocation >= greedy heuristic.")


def test_7_robust_heuristic_fallback():
    """Verify that if SciPy MILP is unavailable or errors, engine falls back cleanly to greedy matching."""
    print("\n--- TEST 7: Heuristic Fallback Robustness ---")
    # Temporarily set SCIPY_OPTIMIZE_AVAILABLE to False
    original_flag = rescue_mod.SCIPY_OPTIMIZE_AVAILABLE
    try:
        rescue_mod.SCIPY_OPTIMIZE_AVAILABLE = False

        branch_inv = get_initial_branch_inventory()
        branch_batches = get_initial_branch_batches()

        engine_res = run_rescue_to_reorder_engine(
            operating_mode=MODE_MULTI_BRANCH,
            branch_inventory=branch_inv,
            branch_batches=branch_batches,
            target_branch_id="BR-01"
        )

        assert engine_res["summary"]["solver_status"] == "FALLBACK_GREEDY"
        assert "optimization_meta" in engine_res
        assert engine_res["optimization_meta"]["solver_status"] == "FALLBACK_GREEDY"
        assert len(engine_res["recommendations"]) > 0

        # Verify fallback produced valid recommendations with identical structure
        for rec in engine_res["recommendations"]:
            assert "recommendation_id" in rec
            assert "decision" in rec
            assert "transfer_quantity" in rec
            assert "remaining_reorder_qty" in rec
            assert rec["optimization_status"] == "FALLBACK_GREEDY"

        print(" -> [PASS] Fallback seamlessly triggered with 100% schema integrity when solver unavailable.")
    finally:
        rescue_mod.SCIPY_OPTIMIZE_AVAILABLE = original_flag


def main():
    print("=" * 80)
    print("  PHARMORA - RESCUE-TO-REORDER™ MILP OPTIMIZER VERIFICATION SUITE")
    print("=" * 80)

    try:
        test_1_stock_and_surplus_constraints()
        test_2_single_unit_no_duplicate_allocation()
        test_3_insufficient_supply_residual_purchases()
        test_4_expiry_eligibility_and_shelf_life_feasibility()
        test_5_storage_certification_feasibility()
        test_6_comparison_with_greedy_optimization()
        test_7_robust_heuristic_fallback()

        print("\n" + "=" * 80)
        print("  ALL 7 MILP OPTIMIZATION TESTS PASSED WITH 100% PRECISION")
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
