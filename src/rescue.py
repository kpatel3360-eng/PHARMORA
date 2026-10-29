"""
PHARMORA Rescue-to-Reorder™ Decision Engine & Multi-Branch / Stock-Sharing Architecture
======================================================================================
Tagline: "Rescue Before You Reorder."

This module implements:
1. Pharmacy Operating Modes:
   - MULTI_BRANCH: Centralized internal inventory balancing across branches.
   - NETWORK_SHARING: Consent-based peer pharmacy stock rescue.
   - SINGLE_STORE: Traditional single-pharmacy local inventory optimization.
2. 7-Step Rescue-to-Reorder Decision Engine:
   Step 1: Check demand and current stock at destination.
   Step 2: Identify eligible surplus stock approaching expiry at source.
   Step 3: Search authorized internal branches or opted-in peer pharmacies.
   Step 4: Evaluate feasibility (medicine identity, shelf-life suitability, storage, compliance).
   Step 5: Estimate remaining requirement after proposed transfer.
   Step 6: Recommend new purchase ONLY for residual shortage.
   Step 7: Provide explicit explanation if no suitable match exists.
3. Multi-Branch Repository & Independent Pharmacy Network Directory.
4. Authorization & Compliance Guards:
   - Strict unexpired batch validation.
   - Zero automatic debit: transfers only execute upon explicit authorized confirmation.
   - Independent inventory boundaries for unrelated pharmacy businesses.
   - Clearly labelled synthetic demo entities.
"""

import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError:
    SCIPY_OPTIMIZE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Operating Mode Constants
# ---------------------------------------------------------------------------
MODE_MULTI_BRANCH = "MULTI_BRANCH"
MODE_NETWORK_SHARING = "NETWORK_SHARING"
MODE_SINGLE_STORE = "SINGLE_STORE"

OPERATING_MODES = {
    MODE_MULTI_BRANCH: {
        "title": "I Manage Multiple Branches",
        "description": "I own or manage multiple pharmacy locations.",
        "icon": "🏢",
        "supporting_text": "Move eligible surplus stock between your branches before purchasing more.",
        "button_label": "Manage My Branches",
        "badge_color": "#166534"
    },
    MODE_NETWORK_SHARING: {
        "title": "I Want to Share Stock With Other Pharmacies",
        "description": "I operate an independent pharmacy and want to connect with other pharmacies.",
        "icon": "🤝",
        "supporting_text": "Discover potential stock-sharing opportunities with participating pharmacies.",
        "button_label": "Explore Pharmacy Network",
        "badge_color": "#0284C7"
    },
    MODE_SINGLE_STORE: {
        "title": "I currently operate just one pharmacy",
        "description": "Single-store inventory optimization with local expiry risk and reorder controls.",
        "icon": "🏪",
        "supporting_text": "Continue with single-store inventory risk detection, with option to enable sharing later.",
        "button_label": "Continue Single Store",
        "badge_color": "#64748B"
    }
}


# ---------------------------------------------------------------------------
# Synthetic Multi-Branch Repository
# ---------------------------------------------------------------------------
DEFAULT_BRANCHES = [
    {
        "branch_id": "BR-01",
        "name": "PHARMORA Central Flagship",
        "location": "Downtown Central Hub, Metro",
        "address": "100 Medical Center Blvd, Suite 101",
        "manager": "Dr. Aris Thorne, PharmD",
        "license_num": "PH-METRO-00812",
        "is_headquarters": True,
        "storage_certified": ["Room Temperature", "Cold Chain 2-8°C"],
        "is_synthetic": True
    },
    {
        "branch_id": "BR-02",
        "name": "PHARMORA North District Clinic",
        "location": "Northside Health Pavilion",
        "address": "450 North Valley Way",
        "manager": "Elena Vance, RPh",
        "license_num": "PH-METRO-00845",
        "is_headquarters": False,
        "storage_certified": ["Room Temperature", "Cold Chain 2-8°C"],
        "is_synthetic": True
    },
    {
        "branch_id": "BR-03",
        "name": "PHARMORA Westside Community Rx",
        "location": "West End Medical Arts",
        "address": "780 Evergreen Parkway",
        "manager": "Marcus Brody, PharmD",
        "license_num": "PH-METRO-00877",
        "is_headquarters": False,
        "storage_certified": ["Room Temperature"],
        "is_synthetic": True
    }
]


# ---------------------------------------------------------------------------
# Synthetic Independent Pharmacy Network Directory
# ---------------------------------------------------------------------------
DEFAULT_NETWORK_PHARMACIES = [
    {
        "pharmacy_id": "NET-SELF",
        "name": "Apex Community Pharmacy (Your Store)",
        "location": "Downtown Metro District",
        "distance_km": 0.0,
        "opted_in": True,
        "sharing_preferences": ["Analgesics", "Anti-inflammatory", "Antibiotics"],
        "privacy_level": "RESTRICTED",  # Masked until mutual request
        "license_verified": True,
        "license_num": "LIC-IND-99104",
        "contact_person": "You (Owner / Pharmacist in Charge)",
        "is_synthetic": True
    },
    {
        "pharmacy_id": "NET-01",
        "name": "Metro Care Pharmacy",
        "location": "Eastside Healthcare Plaza",
        "distance_km": 3.4,
        "opted_in": True,
        "sharing_preferences": ["Anti-inflammatory", "Respiratory", "Analgesics"],
        "privacy_level": "RESTRICTED",
        "license_verified": True,
        "license_num": "LIC-IND-44120",
        "contact_person": "Dr. Sarah Lin, RPh",
        "is_synthetic": True
    },
    {
        "pharmacy_id": "NET-02",
        "name": "Beacon Health Chemists",
        "location": "Midtown Avenue",
        "distance_km": 5.8,
        "opted_in": True,
        "sharing_preferences": ["Respiratory", "Psycholeptics", "Anti-inflammatory"],
        "privacy_level": "RESTRICTED",
        "license_verified": True,
        "license_num": "LIC-IND-77319",
        "contact_person": "Robert Chang, PharmD",
        "is_synthetic": True
    },
    {
        "pharmacy_id": "NET-03",
        "name": "St. Jude Community Rx",
        "location": "South Valley Center",
        "distance_km": 11.2,
        "opted_in": True,
        "sharing_preferences": ["Analgesics", "Antibiotics", "Anti-inflammatory"],
        "privacy_level": "RESTRICTED",
        "license_verified": True,
        "license_num": "LIC-IND-66502",
        "contact_person": "Miriam Davis, RPh",
        "is_synthetic": True
    }
]


def get_initial_branch_inventory() -> pd.DataFrame:
    """
    Returns synthetic, reproducible branch inventory profiles for the 8 baseline medicines.
    Clearly demonstrates:
    - BR-01 needs M01AE (urgent reorder), while BR-02 has surplus M01AE nearing expiry.
    - BR-03 needs R03 (urgent reorder), while BR-01 has surplus R03.
    - BR-02 needs N02BE, but no branch has surplus (forcing standard supplier reorder).
    """
    records = [
        # --- BRANCH 1: Central Flagship ---
        {"branch_id": "BR-01", "medicine_id": "M01AB", "current_stock": 18.0, "pending_orders": 0.0, "safety_stock": 15.0, "lead_time_days": 5, "unit_cost": 8.50, "daily_demand": 5.03},
        {"branch_id": "BR-01", "medicine_id": "M01AE", "current_stock": 12.0, "pending_orders": 0.0, "safety_stock": 12.0, "lead_time_days": 4, "unit_cost": 5.25, "daily_demand": 3.90},  # NEEDS REORDER (33u)
        {"branch_id": "BR-01", "medicine_id": "N02BA", "current_stock": 45.0, "pending_orders": 10.0, "safety_stock": 12.0, "lead_time_days": 3, "unit_cost": 4.10, "daily_demand": 3.88},
        {"branch_id": "BR-01", "medicine_id": "N02BE", "current_stock": 135.0, "pending_orders": 0.0, "safety_stock": 80.0, "lead_time_days": 6, "unit_cost": 6.75, "daily_demand": 29.92}, # NEEDS REORDER (388u)
        {"branch_id": "BR-01", "medicine_id": "N05B",  "current_stock": 120.0, "pending_orders": 0.0, "safety_stock": 25.0, "lead_time_days": 4, "unit_cost": 14.20, "daily_demand": 8.85},
        {"branch_id": "BR-01", "medicine_id": "N05C",  "current_stock": 38.0, "pending_orders": 0.0, "safety_stock": 8.0,  "lead_time_days": 7, "unit_cost": 18.50, "daily_demand": 0.59},
        {"branch_id": "BR-01", "medicine_id": "R03",   "current_stock": 95.0, "pending_orders": 0.0, "safety_stock": 18.0, "lead_time_days": 5, "unit_cost": 12.80, "daily_demand": 5.51},  # SURPLUS (has 35u extra)
        {"branch_id": "BR-01", "medicine_id": "R06",   "current_stock": 35.0, "pending_orders": 5.0,  "safety_stock": 10.0, "lead_time_days": 4, "unit_cost": 7.30, "daily_demand": 2.90},

        # --- BRANCH 2: North District Clinic ---
        {"branch_id": "BR-02", "medicine_id": "M01AB", "current_stock": 22.0, "pending_orders": 0.0, "safety_stock": 10.0, "lead_time_days": 4, "unit_cost": 8.50, "daily_demand": 2.20},
        {"branch_id": "BR-02", "medicine_id": "M01AE", "current_stock": 42.0, "pending_orders": 0.0, "safety_stock": 8.0,  "lead_time_days": 4, "unit_cost": 5.25, "daily_demand": 1.10},  # SURPLUS EXPIRING (has 22u surplus!)
        {"branch_id": "BR-02", "medicine_id": "N02BA", "current_stock": 25.0, "pending_orders": 0.0, "safety_stock": 8.0,  "lead_time_days": 3, "unit_cost": 4.10, "daily_demand": 1.50},
        {"branch_id": "BR-02", "medicine_id": "N02BE", "current_stock": 18.0, "pending_orders": 0.0, "safety_stock": 40.0, "lead_time_days": 5, "unit_cost": 6.75, "daily_demand": 12.00}, # NEEDS REORDER (110u), NO BRANCH HAS SURPLUS
        {"branch_id": "BR-02", "medicine_id": "N05B",  "current_stock": 30.0, "pending_orders": 0.0, "safety_stock": 15.0, "lead_time_days": 4, "unit_cost": 14.20, "daily_demand": 3.20},
        {"branch_id": "BR-02", "medicine_id": "N05C",  "current_stock": 15.0, "pending_orders": 0.0, "safety_stock": 5.0,  "lead_time_days": 6, "unit_cost": 18.50, "daily_demand": 0.30},
        {"branch_id": "BR-02", "medicine_id": "R03",   "current_stock": 12.0, "pending_orders": 0.0, "safety_stock": 12.0, "lead_time_days": 4, "unit_cost": 12.80, "daily_demand": 2.10},
        {"branch_id": "BR-02", "medicine_id": "R06",   "current_stock": 18.0, "pending_orders": 0.0, "safety_stock": 8.0,  "lead_time_days": 3, "unit_cost": 7.30, "daily_demand": 1.00},

        # --- BRANCH 3: Westside Community Rx ---
        {"branch_id": "BR-03", "medicine_id": "M01AB", "current_stock": 14.0, "pending_orders": 0.0, "safety_stock": 8.0,  "lead_time_days": 4, "unit_cost": 8.50, "daily_demand": 1.80},
        {"branch_id": "BR-03", "medicine_id": "M01AE", "current_stock": 16.0, "pending_orders": 0.0, "safety_stock": 8.0,  "lead_time_days": 3, "unit_cost": 5.25, "daily_demand": 1.40},
        {"branch_id": "BR-03", "medicine_id": "N02BA", "current_stock": 20.0, "pending_orders": 0.0, "safety_stock": 8.0,  "lead_time_days": 3, "unit_cost": 4.10, "daily_demand": 1.20},
        {"branch_id": "BR-03", "medicine_id": "N02BE", "current_stock": 25.0, "pending_orders": 0.0, "safety_stock": 35.0, "lead_time_days": 5, "unit_cost": 6.75, "daily_demand": 8.50},
        {"branch_id": "BR-03", "medicine_id": "N05B",  "current_stock": 22.0, "pending_orders": 0.0, "safety_stock": 12.0, "lead_time_days": 4, "unit_cost": 14.20, "daily_demand": 2.40},
        {"branch_id": "BR-03", "medicine_id": "N05C",  "current_stock": 10.0, "pending_orders": 0.0, "safety_stock": 4.0,  "lead_time_days": 5, "unit_cost": 18.50, "daily_demand": 0.20},
        {"branch_id": "BR-03", "medicine_id": "R03",   "current_stock": 8.0,  "pending_orders": 0.0, "safety_stock": 15.0, "lead_time_days": 4, "unit_cost": 12.80, "daily_demand": 3.80},  # NEEDS REORDER (40u), CAN BE RESCUED FROM BR-01!
        {"branch_id": "BR-03", "medicine_id": "R06",   "current_stock": 14.0, "pending_orders": 0.0, "safety_stock": 6.0,  "lead_time_days": 3, "unit_cost": 7.30, "daily_demand": 0.90},
    ]
    df = pd.DataFrame(records)
    df["is_synthetic"] = True
    return df


def get_initial_branch_batches(reference_date: datetime.date = datetime.date(2019, 10, 9)) -> pd.DataFrame:
    """
    Returns synthetic, reproducible branch batch records with expiry dates.
    All batches are strictly evaluated for expiry eligibility.
    """
    ref_str = reference_date.strftime("%Y-%m-%d")
    batches = [
        # BR-01 Batches
        {"batch_id": "B-M01AB-BR1-01", "branch_id": "BR-01", "medicine_id": "M01AB", "quantity": 18.0, "expiry_date": "2020-05-15", "unit_cost": 8.50, "storage_condition": "Room Temperature"},
        {"batch_id": "B-M01AE-BR1-01", "branch_id": "BR-01", "medicine_id": "M01AE", "quantity": 12.0, "expiry_date": "2020-03-20", "unit_cost": 5.25, "storage_condition": "Room Temperature"},
        {"batch_id": "B-N02BA-BR1-01", "branch_id": "BR-01", "medicine_id": "N02BA", "quantity": 45.0, "expiry_date": "2020-08-10", "unit_cost": 4.10, "storage_condition": "Room Temperature"},
        {"batch_id": "B-N02BE-BR1-01", "branch_id": "BR-01", "medicine_id": "N02BE", "quantity": 135.0, "expiry_date": "2020-04-12", "unit_cost": 6.75, "storage_condition": "Room Temperature"},
        {"batch_id": "B-N05B-BR1-01",  "branch_id": "BR-01", "medicine_id": "N05B",  "quantity": 120.0, "expiry_date": "2020-09-30", "unit_cost": 14.20, "storage_condition": "Room Temperature"},
        {"batch_id": "B-N05C-BR1-01",  "branch_id": "BR-01", "medicine_id": "N05C",  "quantity": 38.0,  "expiry_date": "2020-11-15", "unit_cost": 18.50, "storage_condition": "Room Temperature"},
        # BR-01 R03 has a surplus batch expiring in 65 days:
        {"batch_id": "B-R03-BR1-01",   "branch_id": "BR-01", "medicine_id": "R03",   "quantity": 60.0, "expiry_date": "2020-07-20", "unit_cost": 12.80, "storage_condition": "Room Temperature"},
        {"batch_id": "B-R03-BR1-02",   "branch_id": "BR-01", "medicine_id": "R03",   "quantity": 35.0, "expiry_date": "2019-12-15", "unit_cost": 12.80, "storage_condition": "Room Temperature"}, # SURPLUS BATCH (67 days left)
        {"batch_id": "B-R06-BR1-01",   "branch_id": "BR-01", "medicine_id": "R06",   "quantity": 35.0, "expiry_date": "2020-06-25", "unit_cost": 7.30, "storage_condition": "Room Temperature"},

        # BR-02 Batches (North District)
        {"batch_id": "B-M01AB-BR2-01", "branch_id": "BR-02", "medicine_id": "M01AB", "quantity": 22.0, "expiry_date": "2020-04-10", "unit_cost": 8.50, "storage_condition": "Room Temperature"},
        # BR-02 M01AE has a surplus batch nearing expiry (48 days left):
        {"batch_id": "B-M01AE-BR2-01", "branch_id": "BR-02", "medicine_id": "M01AE", "quantity": 25.0, "expiry_date": "2019-11-26", "unit_cost": 5.25, "storage_condition": "Room Temperature"}, # SURPLUS BATCH (48 days left)
        {"batch_id": "B-M01AE-BR2-02", "branch_id": "BR-02", "medicine_id": "M01AE", "quantity": 17.0, "expiry_date": "2020-05-30", "unit_cost": 5.25, "storage_condition": "Room Temperature"},
        {"batch_id": "B-N02BE-BR2-01", "branch_id": "BR-02", "medicine_id": "N02BE", "quantity": 18.0, "expiry_date": "2020-02-14", "unit_cost": 6.75, "storage_condition": "Room Temperature"},

        # BR-03 Batches (Westside Community)
        {"batch_id": "B-M01AB-BR3-01", "branch_id": "BR-03", "medicine_id": "M01AB", "quantity": 14.0, "expiry_date": "2020-07-01", "unit_cost": 8.50, "storage_condition": "Room Temperature"},
        {"batch_id": "B-M01AE-BR3-01", "branch_id": "BR-03", "medicine_id": "M01AE", "quantity": 16.0, "expiry_date": "2020-06-15", "unit_cost": 5.25, "storage_condition": "Room Temperature"},
        {"batch_id": "B-R03-BR3-01",   "branch_id": "BR-03", "medicine_id": "R03",   "quantity": 8.0,  "expiry_date": "2020-04-05", "unit_cost": 12.80, "storage_condition": "Room Temperature"},
    ]
    df = pd.DataFrame(batches)
    df["expiry_dt"] = pd.to_datetime(df["expiry_date"])
    ref_dt = pd.to_datetime(ref_str)
    df["days_to_expiry"] = (df["expiry_dt"] - ref_dt).dt.days
    df["is_expired"] = df["days_to_expiry"] <= 0
    df["is_synthetic"] = True
    return df


def get_initial_network_stock(reference_date: datetime.date = datetime.date(2019, 10, 9)) -> pd.DataFrame:
    """
    Returns synthetic surplus stock made discoverable by opted-in network peer pharmacies.
    Information is consent-governed and privacy-masked until mutual connection request.
    """
    ref_dt = pd.to_datetime(reference_date)
    records = [
        {
            "pharmacy_id": "NET-01",
            "name": "Metro Care Pharmacy",
            "pharmacy_name": "Metro Care Pharmacy",
            "medicine_id": "M01AE",
            "medicine_name": "M01AE - Ibuprofen",
            "available_quantity": 25.0,
            "expiry_date": "2019-12-10",
            "unit_cost": 5.25,
            "distance_km": 3.4,
            "sharing_status": "AVAILABLE_FOR_RESCUE",
            "storage_certified": True,
            "is_synthetic": True
        },
        {
            "pharmacy_id": "NET-02",
            "name": "Beacon Health Chemists",
            "pharmacy_name": "Beacon Health Chemists",
            "medicine_id": "R03",
            "medicine_name": "R03 - Anti-asthmatic Inhalers",
            "available_quantity": 30.0,
            "expiry_date": "2019-12-25",
            "unit_cost": 12.80,
            "distance_km": 5.8,
            "sharing_status": "AVAILABLE_FOR_RESCUE",
            "storage_certified": True,
            "is_synthetic": True
        },
        {
            "pharmacy_id": "NET-03",
            "name": "St. Jude Community Rx",
            "pharmacy_name": "St. Jude Community Rx",
            "medicine_id": "N02BA",
            "medicine_name": "N02BA - Aspirin",
            "available_quantity": 40.0,
            "expiry_date": "2020-01-15",
            "unit_cost": 4.10,
            "distance_km": 11.2,
            "sharing_status": "AVAILABLE_FOR_RESCUE",
            "storage_certified": True,
            "is_synthetic": True
        }
    ]
    df = pd.DataFrame(records)
    df["expiry_dt"] = pd.to_datetime(df["expiry_date"])
    df["days_to_expiry"] = (df["expiry_dt"] - ref_dt).dt.days
    return df


# ---------------------------------------------------------------------------
# The 7-Step Rescue-to-Reorder™ Decision Engine
# ---------------------------------------------------------------------------
def calculate_branch_reorder_requirement(
    current_stock: float,
    pending_orders: float,
    daily_demand: float,
    lead_time_days: int,
    safety_stock: float,
    review_period_days: int = 7
) -> Dict[str, Any]:
    """
    Step 1: Check demand and current stock to determine reorder point and order need.
    """
    inventory_position = current_stock + pending_orders
    lead_time_demand = daily_demand * lead_time_days
    reorder_point = lead_time_demand + safety_stock
    target_stock = daily_demand * (lead_time_days + review_period_days) + safety_stock

    if inventory_position <= reorder_point:
        gross_reorder = max(0.0, math_ceil(target_stock - inventory_position))
        is_needed = gross_reorder > 0
    else:
        gross_reorder = 0.0
        is_needed = False

    return {
        "inventory_position": round(inventory_position, 1),
        "lead_time_demand": round(lead_time_demand, 1),
        "reorder_point": round(reorder_point, 1),
        "target_stock": round(target_stock, 1),
        "gross_reorder_qty": int(gross_reorder),
        "is_reorder_needed": is_needed
    }


def math_ceil(val: float) -> int:
    return int(np.ceil(val))


def optimize_rescue_transfers(
    destinations: List[Dict[str, Any]],
    candidate_batches: List[Dict[str, Any]],
    source_surpluses: Dict[Tuple[str, str], float],
    storage_cert_map: Dict[str, List[str]],
    operating_mode: str = MODE_MULTI_BRANCH,
    logistics_rate_internal: float = 0.75,
    reference_date: datetime.date = datetime.date(2019, 10, 9)
) -> Dict[str, Any]:
    """
    Executes exact constraint-based cost optimization for multi-branch/network stock transfers:
    - Replaces greedy first-fit heuristics with Mixed-Integer Linear Programming (MILP) via SciPy HiGHS.
    - Minimizes: Total Cost = Logistics Shipping + Residual Purchases + Projected Expiry Waste Loss.
    - Constraints Enforced:
      1. Demand Fulfillment: Total transfers of drug m to dest j <= Gross Reorder Required (R_j).
      2. No Duplicate Allocation: Total transfers of batch b across all destinations <= Batch physical quantity (Q_b).
      3. Aggregate Source Surplus: Total transfers of drug m out of branch s <= Net branch surplus (S_s,m).
      4. Shelf-Life Feasibility: Destination can consume transferred units before batch expiry (X_b,j <= floor(d_j * D_b)).
      5. Expiry Eligibility: Days to expiry D_b > 14 days (GxP safety quarantine buffer).
      6. Storage Compatibility: Destination branch storage certification contains batch storage condition.
      7. Integrality: Transfer quantities are strictly non-negative integers.
    """
    # 1. Identify all feasible transfer candidate arcs
    arcs = []
    for j_idx, dest in enumerate(destinations):
        dest_id = dest["dest_id"]
        med_id = dest["medicine_id"]
        gross_req = int(dest["gross_reorder_qty"])
        if gross_req <= 0:
            continue
        dest_daily = float(dest.get("daily_demand", 1.0))
        dest_certs = storage_cert_map.get(dest_id, ["Room Temperature", "Cold Chain 2-8°C"])

        for b_idx, batch in enumerate(candidate_batches):
            if batch["medicine_id"] != med_id:
                continue
            if batch["source_id"] == dest_id:
                continue
            days_left = int(batch.get("days_to_expiry", 0))
            if days_left <= 14:  # GxP safety: near expiry or expired
                continue

            # Storage compatibility constraint
            batch_storage = batch.get("storage_condition", "Room Temperature")
            if batch_storage not in dest_certs:
                continue

            # Consumption rate constraint: destination must be able to consume at least 1 unit before expiry
            max_consumable = int(np.floor(days_left * dest_daily)) if dest_daily > 0 else 0
            if max_consumable < 1:
                continue

            b_qty = float(batch.get("quantity", 0.0))
            s_surplus = float(source_surpluses.get((batch["source_id"], med_id), 0.0))

            ub_k = min(b_qty, s_surplus, float(gross_req), float(max_consumable))
            if ub_k < 1.0:
                continue

            dist = float(batch.get("distance_km", 0.0))
            if operating_mode == MODE_NETWORK_SHARING:
                unit_logistics = max(1.0, round(0.20 * dist, 2))
            else:
                unit_logistics = logistics_rate_internal

            arcs.append({
                "arc_id": len(arcs),
                "dest_idx": j_idx,
                "dest_id": dest_id,
                "dest_name": dest["dest_name"],
                "batch_idx": b_idx,
                "batch_id": batch["batch_id"],
                "source_id": batch["source_id"],
                "source_name": batch["source_name"],
                "medicine_id": med_id,
                "batch_expiry": batch["expiry_date"],
                "days_to_expiry": days_left,
                "unit_cost": float(dest["unit_cost"]),
                "unit_logistics": unit_logistics,
                "ub": int(np.floor(ub_k)),
                "distance_km": dist
            })

    if not arcs:
        return {
            "allocations": {},
            "solver_status": "NO_FEASIBLE_TRANSFERS",
            "method": "Constraint Evaluation (No Eligible Surplus Arcs)",
            "arcs": [],
            "total_cost": 0.0,
            "logistics_cost": 0.0,
            "rescued_units": 0
        }

    K = len(arcs)

    # 2. Objective coefficients:
    # Net cost per transferred unit = unit_logistics - (wholesale_saved + waste_avoided) + tie-breakers
    c_vec = np.zeros(K)
    for k, a in enumerate(arcs):
        c_m = a["unit_cost"]
        # Minimizing this rewards rescuing units, penalizes logistics, and breaks ties by FEFO shelf life
        c_vec[k] = a["unit_logistics"] - (2.0 * c_m) + (0.0001 * a["days_to_expiry"]) + (0.001 * a["distance_km"])

    bounds = Bounds(lb=np.zeros(K), ub=np.array([a["ub"] for a in arcs], dtype=float))
    integrality = np.ones(K)

    # 3. Linear constraints
    constraint_rows = []
    b_u_list = []

    # Row group A: Destination demand caps (sum of transfers to dest <= gross_req)
    for j_idx, dest in enumerate(destinations):
        matching_arcs = [k for k, a in enumerate(arcs) if a["dest_idx"] == j_idx]
        if matching_arcs:
            row = np.zeros(K)
            for k in matching_arcs:
                row[k] = 1.0
            constraint_rows.append(row)
            b_u_list.append(float(dest["gross_reorder_qty"]))

    # Row group B: Batch capacity caps (sum of transfers of batch across all destinations <= batch quantity)
    distinct_batches = sorted(list({a["batch_id"] for a in arcs}))
    for b_id in distinct_batches:
        matching_arcs = [k for k, a in enumerate(arcs) if a["batch_id"] == b_id]
        if matching_arcs:
            row = np.zeros(K)
            for k in matching_arcs:
                row[k] = 1.0
            constraint_rows.append(row)
            batch_obj = next(b for b in candidate_batches if b["batch_id"] == b_id)
            b_u_list.append(float(batch_obj["quantity"]))

    # Row group C: Source surplus caps (sum of transfers out of (source, med) <= net branch surplus)
    distinct_sources = sorted(list({(a["source_id"], a["medicine_id"]) for a in arcs}))
    for s_id, m_id in distinct_sources:
        matching_arcs = [k for k, a in enumerate(arcs) if a["source_id"] == s_id and a["medicine_id"] == m_id]
        if matching_arcs:
            row = np.zeros(K)
            for k in matching_arcs:
                row[k] = 1.0
            constraint_rows.append(row)
            b_u_list.append(float(source_surpluses.get((s_id, m_id), 0.0)))

    A_mat = np.array(constraint_rows)
    b_u_arr = np.array(b_u_list)

    allocations = {}
    solver_status = "FALLBACK_GREEDY"

    if SCIPY_OPTIMIZE_AVAILABLE:
        try:
            constraints = LinearConstraint(A_mat, -np.inf, b_u_arr)
            res = milp(c=c_vec, integrality=integrality, bounds=bounds, constraints=constraints)
            if res.success:
                allocations = {k: int(np.round(res.x[k])) for k in range(K) if np.round(res.x[k]) > 0}
                solver_status = "OPTIMAL_MILP"
            else:
                solver_status = "FALLBACK_GREEDY"
        except Exception:
            solver_status = "FALLBACK_GREEDY"

    tot_rescued = sum(allocations.values())
    tot_logistics = sum(allocations[k] * arcs[k]["unit_logistics"] for k in allocations)

    return {
        "allocations": allocations,
        "solver_status": solver_status,
        "method": "SciPy MILP (HiGHS Exact Constraint Solver)" if solver_status == "OPTIMAL_MILP" else "Greedy Heuristic Fallback",
        "arcs": arcs,
        "logistics_cost": round(tot_logistics, 2),
        "rescued_units": int(tot_rescued)
    }


def _greedy_rescue_match_fallback(
    destinations: List[Dict[str, Any]],
    candidate_batches: List[Dict[str, Any]],
    source_surpluses: Dict[Tuple[str, str], float],
    storage_cert_map: Dict[str, List[str]],
    operating_mode: str = MODE_MULTI_BRANCH,
    logistics_rate_internal: float = 0.75
) -> Tuple[List[Dict[str, Any]], float, float, float, float, float]:
    """
    Deterministic first-fit greedy fallback executed if the MILP solver is unavailable or fails.
    Ensures uninterrupted operational resilience.
    """
    recommendations = []
    tot_rescued = 0.0
    tot_waste_avoided = 0.0
    tot_purchases_avoided = 0.0
    tot_residuals = 0.0
    tot_logistics = 0.0

    working_surplus = dict(source_surpluses)
    working_batch_qty = {b["batch_id"]: float(b["quantity"]) for b in candidate_batches}

    for dest in destinations:
        dest_id = dest["dest_id"]
        med_id = dest["medicine_id"]
        gross_qty = int(dest["gross_reorder_qty"])
        unit_cost = float(dest["unit_cost"])
        dest_daily = float(dest.get("daily_demand", 1.0))
        dest_certs = storage_cert_map.get(dest_id, ["Room Temperature", "Cold Chain 2-8°C"])

        best_match = None
        for b in sorted(candidate_batches, key=lambda x: x.get("days_to_expiry", 999)):
            if b["medicine_id"] != med_id or b["source_id"] == dest_id:
                continue
            days_left = b["days_to_expiry"]
            if days_left <= 14:
                continue
            if b.get("storage_condition", "Room Temperature") not in dest_certs:
                continue

            needed_days = min(working_surplus.get((b["source_id"], med_id), 0.0), gross_qty) / dest_daily if dest_daily > 0 else 999
            if days_left < needed_days:
                continue

            avail_s = working_surplus.get((b["source_id"], med_id), 0.0)
            avail_b = working_batch_qty.get(b["batch_id"], 0.0)
            t_qty = min(avail_s, float(gross_qty), avail_b)
            if t_qty >= 1.0:
                transfer_qty_int = int(np.floor(t_qty))
                dist = float(b.get("distance_km", 0.0))
                unit_log = max(1.0, round(0.20 * dist, 2)) if operating_mode == MODE_NETWORK_SHARING else logistics_rate_internal
                best_match = {
                    "source_id": b["source_id"],
                    "source_name": b["source_name"],
                    "batch_id": b["batch_id"],
                    "days_to_expiry": days_left,
                    "expiry_date": b["expiry_date"],
                    "transfer_qty": transfer_qty_int,
                    "unit_cost": unit_cost,
                    "unit_logistics": unit_log
                }
                working_surplus[(b["source_id"], med_id)] -= transfer_qty_int
                working_batch_qty[b["batch_id"]] -= transfer_qty_int
                break

        if best_match:
            r_qty = best_match["transfer_qty"]
            rem_order = max(0, gross_qty - r_qty)
            w_avoided = round(r_qty * unit_cost, 2)
            log_fee = round(r_qty * best_match["unit_logistics"], 2)

            tot_rescued += r_qty
            tot_waste_avoided += w_avoided
            tot_purchases_avoided += w_avoided
            tot_residuals += rem_order
            tot_logistics += log_fee

            recommendations.append({
                "recommendation_id": f"REC-MB-FALLBACK-{dest_id}-{med_id}",
                "operating_mode": operating_mode,
                "destination_id": dest_id,
                "destination_name": dest["dest_name"],
                "source_id": best_match["source_id"],
                "source_name": best_match["source_name"],
                "medicine_id": med_id,
                "batch_id": best_match["batch_id"],
                "batch_expiry": best_match["expiry_date"],
                "days_to_expiry": best_match["days_to_expiry"],
                "transfer_quantity": r_qty,
                "original_reorder_qty": gross_qty,
                "remaining_reorder_qty": rem_order,
                "unit_cost": unit_cost,
                "financial_waste_avoided": w_avoided,
                "purchases_avoided": w_avoided,
                "logistics_cost": log_fee,
                "decision": "RECOMMENDED_TRANSFER" if operating_mode == MODE_MULTI_BRANCH else "POTENTIAL_NETWORK_MATCH",
                "status": "RECOMMENDED" if operating_mode == MODE_MULTI_BRANCH else "POTENTIAL_MATCH",
                "reason": f"Fallback heuristic selected {r_qty} units from {best_match['source_id']} expiring in {best_match['days_to_expiry']} days.",
                "action_label": f"Transfer {r_qty} units from {best_match['source_id']}",
                "residual_action": f"Purchase remaining {rem_order} units from supplier" if rem_order > 0 else "Zero external purchase required (100% Rescued!)",
                "optimization_status": "FALLBACK_GREEDY"
            })
        else:
            tot_residuals += gross_qty
            recommendations.append({
                "recommendation_id": f"REC-EXT-FALLBACK-{dest_id}-{med_id}",
                "operating_mode": operating_mode,
                "destination_id": dest_id,
                "destination_name": dest["dest_name"],
                "source_id": "EXTERNAL_SUPPLIER",
                "source_name": "Authorized External Distributor",
                "medicine_id": med_id,
                "batch_id": "NEW_ORDER",
                "batch_expiry": "Standard Manufacturer Shelf Life",
                "days_to_expiry": 365,
                "transfer_quantity": 0,
                "original_reorder_qty": gross_qty,
                "remaining_reorder_qty": gross_qty,
                "unit_cost": unit_cost,
                "financial_waste_avoided": 0.0,
                "purchases_avoided": 0.0,
                "logistics_cost": 0.0,
                "decision": "SUPPLIER_PURCHASE_ONLY",
                "status": "DIRECT_REORDER",
                "reason": f"Fallback heuristic: No eligible internal branch has surplus stock of {med_id}.",
                "action_label": f"Reorder {gross_qty} units from supplier",
                "residual_action": f"Purchase {gross_qty} units from supplier",
                "optimization_status": "FALLBACK_GREEDY"
            })

    return recommendations, tot_rescued, tot_waste_avoided, tot_purchases_avoided, tot_residuals, tot_logistics


def run_rescue_to_reorder_engine(
    operating_mode: str,
    branch_inventory: pd.DataFrame,
    branch_batches: pd.DataFrame,
    target_branch_id: Optional[str] = None,
    network_stock: Optional[pd.DataFrame] = None,
    active_confirmed_transfers: Optional[List[Dict[str, Any]]] = None,
    reference_date: datetime.date = datetime.date(2019, 10, 9)
) -> Dict[str, Any]:
    """
    Executes the 7-Step Rescue-to-Reorder decision algorithm powered by exact MILP constraint optimization:
    
    1. Check demand and current stock at destination branches.
    2. Identify eligible surplus stock approaching expiry at other locations.
    3. Search authorized internal branches or opted-in peer pharmacies.
    4. Evaluate feasibility constraints (medicine identity, unexpired, shelf-life suitability, storage certification).
    5. Formulate and solve the global minimum-cost multi-branch transfer problem using SciPy MILP.
    6. Recommend eligible transfers before purchasing new stock, then calculate residual purchase requirements.
    7. Provide explicit explanation for why each transfer was selected or why external procurement is needed.
    
    Returns a comprehensive dictionary with:
    - recommendations: List of structured transfer opportunities
    - summary: Aggregated metrics (total rescued, waste avoided, purchases avoided, logistics costs, solver status)
    - comparison: 'Transfer First' vs 'Purchase New' ledger
    - optimization_meta: Solver status and constraint dimensions
    """
    recommendations = []
    tot_rescued_units = 0.0
    tot_waste_avoided_val = 0.0
    tot_purchases_avoided_val = 0.0
    tot_residual_purchases_units = 0.0
    tot_gross_reorder_units = 0.0
    tot_logistics_cost = 0.0

    if active_confirmed_transfers is None:
        active_confirmed_transfers = []

    # Map of already reserved/transferred quantities by batch or branch
    reserved_by_source = {}
    for t in active_confirmed_transfers:
        if t.get("status") in ["CONFIRMED", "ACCEPTED", "IN_TRANSIT"]:
            k = (t["source_id"], t["medicine_id"])
            reserved_by_source[k] = reserved_by_source.get(k, 0.0) + t["quantity"]

    ref_dt = pd.to_datetime(reference_date)

    # Storage certifications map
    storage_cert_map = {b["branch_id"]: b.get("storage_certified", ["Room Temperature"]) for b in DEFAULT_BRANCHES}
    storage_cert_map["NET-SELF"] = ["Room Temperature", "Cold Chain 2-8°C"]
    storage_cert_map["SINGLE_STORE"] = ["Room Temperature", "Cold Chain 2-8°C"]

    solver_status = "OPTIMAL_MILP"
    optimization_method = "SciPy MILP (HiGHS Exact Constraint Solver)"

    if operating_mode == MODE_MULTI_BRANCH:
        # Multi-Branch internal transfer engine
        branches_to_eval = [target_branch_id] if target_branch_id else branch_inventory["branch_id"].unique().tolist()

        # Step 1: Collect all destination requirements
        destinations = []
        for b_id in branches_to_eval:
            b_inv = branch_inventory[branch_inventory["branch_id"] == b_id]
            for _, row in b_inv.iterrows():
                med_id = row["medicine_id"]
                req = calculate_branch_reorder_requirement(
                    current_stock=row["current_stock"],
                    pending_orders=row["pending_orders"],
                    daily_demand=row["daily_demand"],
                    lead_time_days=int(row["lead_time_days"]),
                    safety_stock=row["safety_stock"]
                )
                gross_qty = req["gross_reorder_qty"]
                tot_gross_reorder_units += gross_qty

                if req["is_reorder_needed"]:
                    destinations.append({
                        "dest_id": b_id,
                        "dest_name": next((b["name"] for b in DEFAULT_BRANCHES if b["branch_id"] == b_id), b_id),
                        "medicine_id": med_id,
                        "gross_reorder_qty": gross_qty,
                        "daily_demand": float(row["daily_demand"]),
                        "unit_cost": float(row["unit_cost"]),
                        "storage_certified": storage_cert_map.get(b_id, ["Room Temperature", "Cold Chain 2-8°C"])
                    })

        # Step 2: Compute available surplus at all potential source branches
        source_surpluses = {}
        for s_id in branch_inventory["branch_id"].unique():
            s_inv = branch_inventory[branch_inventory["branch_id"] == s_id]
            for _, s_row in s_inv.iterrows():
                med_id = s_row["medicine_id"]
                s_demand = s_row["daily_demand"]
                s_lt = s_row["lead_time_days"]
                s_safety = s_row["safety_stock"]
                s_needed_local = (s_demand * (s_lt + 14)) + s_safety
                raw_surplus = max(0.0, s_row["current_stock"] - s_needed_local)
                already_committed = reserved_by_source.get((s_id, med_id), 0.0)
                source_surpluses[(s_id, med_id)] = max(0.0, raw_surplus - already_committed)

        # Step 3: Collect candidate unexpired batches from sources
        candidate_batches = []
        for _, b_row in branch_batches.iterrows():
            if b_row["is_expired"] or b_row["days_to_expiry"] <= 14:
                continue
            s_id = b_row["branch_id"]
            med_id = b_row["medicine_id"]
            if source_surpluses.get((s_id, med_id), 0.0) < 0.5:
                continue

            candidate_batches.append({
                "batch_id": b_row["batch_id"],
                "source_id": s_id,
                "source_name": next((b["name"] for b in DEFAULT_BRANCHES if b["branch_id"] == s_id), s_id),
                "medicine_id": med_id,
                "quantity": float(b_row["quantity"]),
                "expiry_date": str(b_row["expiry_date"]),
                "days_to_expiry": int(b_row["days_to_expiry"]),
                "unit_cost": float(b_row.get("unit_cost", 0.0)),
                "storage_condition": b_row.get("storage_condition", "Room Temperature"),
                "distance_km": 0.0
            })

        # Step 4 & 5: Run MILP optimization
        opt_res = optimize_rescue_transfers(
            destinations=destinations,
            candidate_batches=candidate_batches,
            source_surpluses=source_surpluses,
            storage_cert_map=storage_cert_map,
            operating_mode=MODE_MULTI_BRANCH,
            logistics_rate_internal=0.75,
            reference_date=reference_date
        )

        solver_status = opt_res["solver_status"]
        optimization_method = opt_res["method"]

        if solver_status == "FALLBACK_GREEDY":
            # Safe Fallback to deterministic greedy matching
            f_recs, f_res, f_wa, f_pa, f_rem, f_log = _greedy_rescue_match_fallback(
                destinations=destinations,
                candidate_batches=candidate_batches,
                source_surpluses=source_surpluses,
                storage_cert_map=storage_cert_map,
                operating_mode=MODE_MULTI_BRANCH,
                logistics_rate_internal=0.75
            )
            recommendations.extend(f_recs)
            tot_rescued_units += f_res
            tot_waste_avoided_val += f_wa
            tot_purchases_avoided_val += f_pa
            tot_residual_purchases_units += f_rem
            tot_logistics_cost += f_log
        else:
            # Build structured recommendations from MILP allocations
            allocations = opt_res["allocations"]
            arcs = opt_res["arcs"]

            for j_idx, dest in enumerate(destinations):
                dest_id = dest["dest_id"]
                med_id = dest["medicine_id"]
                gross_qty = dest["gross_reorder_qty"]
                unit_cost = dest["unit_cost"]

                matching_alloc_arcs = [
                    (arcs[k], allocations[k]) for k in allocations if arcs[k]["dest_idx"] == j_idx
                ]
                total_transferred_for_dest = sum(t_qty for _, t_qty in matching_alloc_arcs)
                remaining_reorder = max(0, gross_qty - total_transferred_for_dest)

                if matching_alloc_arcs:
                    for arc, t_qty in matching_alloc_arcs:
                        waste_avoided = round(t_qty * unit_cost, 2)
                        log_fee = round(t_qty * arc["unit_logistics"], 2)

                        tot_rescued_units += t_qty
                        tot_waste_avoided_val += waste_avoided
                        tot_purchases_avoided_val += waste_avoided
                        tot_logistics_cost += log_fee

                        recommendations.append({
                            "recommendation_id": f"REC-MB-{dest_id}-{med_id}-{arc['batch_id']}",
                            "operating_mode": MODE_MULTI_BRANCH,
                            "destination_id": dest_id,
                            "destination_name": dest["dest_name"],
                            "source_id": arc["source_id"],
                            "source_name": arc["source_name"],
                            "medicine_id": med_id,
                            "batch_id": arc["batch_id"],
                            "batch_expiry": arc["batch_expiry"],
                            "days_to_expiry": arc["days_to_expiry"],
                            "transfer_quantity": t_qty,
                            "original_reorder_qty": gross_qty,
                            "remaining_reorder_qty": remaining_reorder,
                            "unit_cost": unit_cost,
                            "financial_waste_avoided": waste_avoided,
                            "purchases_avoided": waste_avoided,
                            "logistics_cost": log_fee,
                            "decision": "RECOMMENDED_TRANSFER",
                            "status": "RECOMMENDED",
                            "reason": (
                                f"Constraint optimizer allocated {t_qty} units from {arc['batch_id']} at {arc['source_name']} "
                                f"({arc['days_to_expiry']} days shelf-life). Satisfies storage and destination consumption rate "
                                f"({dest['daily_demand']:.1f}/day). Estimated logistics fee: ₹{log_fee:.2f}."
                            ),
                            "action_label": f"Transfer {t_qty} units from {arc['source_id']}",
                            "residual_action": (
                                f"Purchase remaining {remaining_reorder} units from supplier"
                                if remaining_reorder > 0 else "Zero external purchase required (100% Rescued!)"
                            ),
                            "optimization_status": solver_status
                        })
                    tot_residual_purchases_units += remaining_reorder
                else:
                    tot_residual_purchases_units += gross_qty
                    recommendations.append({
                        "recommendation_id": f"REC-EXT-{dest_id}-{med_id}",
                        "operating_mode": MODE_MULTI_BRANCH,
                        "destination_id": dest_id,
                        "destination_name": dest["dest_name"],
                        "source_id": "EXTERNAL_SUPPLIER",
                        "source_name": "Authorized External Distributor",
                        "medicine_id": med_id,
                        "batch_id": "NEW_ORDER",
                        "batch_expiry": "Standard Manufacturer Shelf Life",
                        "days_to_expiry": 365,
                        "transfer_quantity": 0,
                        "original_reorder_qty": gross_qty,
                        "remaining_reorder_qty": gross_qty,
                        "unit_cost": unit_cost,
                        "financial_waste_avoided": 0.0,
                        "purchases_avoided": 0.0,
                        "logistics_cost": 0.0,
                        "decision": "SUPPLIER_PURCHASE_ONLY",
                        "status": "DIRECT_REORDER",
                        "reason": (
                            f"Constraint solver verified: No eligible internal branch has surplus stock of {med_id} matching "
                            f"safety buffer, shelf-life, and storage compatibility constraints. External supplier purchase recommended."
                        ),
                        "action_label": f"Reorder {gross_qty} units from supplier",
                        "residual_action": f"Purchase {gross_qty} units from supplier",
                        "optimization_status": solver_status
                    })

    elif operating_mode == MODE_NETWORK_SHARING:
        # Independent Pharmacy Network Sharing Engine
        if network_stock is None or network_stock.empty:
            network_stock = get_initial_network_stock(reference_date)

        b1_inv = branch_inventory[branch_inventory["branch_id"] == "BR-01"]
        destinations = []
        for _, row in b1_inv.iterrows():
            med_id = row["medicine_id"]
            req = calculate_branch_reorder_requirement(
                current_stock=row["current_stock"],
                pending_orders=row["pending_orders"],
                daily_demand=row["daily_demand"],
                lead_time_days=int(row["lead_time_days"]),
                safety_stock=row["safety_stock"]
            )
            gross_qty = req["gross_reorder_qty"]
            tot_gross_reorder_units += gross_qty
            if req["is_reorder_needed"]:
                destinations.append({
                    "dest_id": "NET-SELF",
                    "dest_name": "Apex Community Pharmacy (Your Store)",
                    "medicine_id": med_id,
                    "gross_reorder_qty": gross_qty,
                    "daily_demand": float(row["daily_demand"]),
                    "unit_cost": float(row["unit_cost"]),
                    "storage_certified": storage_cert_map.get("NET-SELF", ["Room Temperature", "Cold Chain 2-8°C"])
                })

        # Candidate peer batches
        candidate_batches = []
        source_surpluses = {}
        for _, peer in network_stock.iterrows():
            p_id = peer["pharmacy_id"]
            m_id = peer["medicine_id"]
            qty = float(peer["available_quantity"])
            source_surpluses[(p_id, m_id)] = qty
            candidate_batches.append({
                "batch_id": f"LOT-{p_id}-{m_id}",
                "source_id": p_id,
                "source_name": f"{peer['name']} ({peer['distance_km']:.1f} km away)",
                "medicine_id": m_id,
                "quantity": qty,
                "expiry_date": peer["expiry_date"],
                "days_to_expiry": int(peer["days_to_expiry"]),
                "unit_cost": float(peer.get("unit_cost", 5.0)),
                "storage_condition": "Room Temperature",
                "distance_km": float(peer["distance_km"])
            })

        opt_res = optimize_rescue_transfers(
            destinations=destinations,
            candidate_batches=candidate_batches,
            source_surpluses=source_surpluses,
            storage_cert_map=storage_cert_map,
            operating_mode=MODE_NETWORK_SHARING,
            reference_date=reference_date
        )

        solver_status = opt_res["solver_status"]
        optimization_method = opt_res["method"]

        if solver_status == "FALLBACK_GREEDY":
            f_recs, f_res, f_wa, f_pa, f_rem, f_log = _greedy_rescue_match_fallback(
                destinations=destinations,
                candidate_batches=candidate_batches,
                source_surpluses=source_surpluses,
                storage_cert_map=storage_cert_map,
                operating_mode=MODE_NETWORK_SHARING
            )
            recommendations.extend(f_recs)
            tot_rescued_units += f_res
            tot_waste_avoided_val += f_wa
            tot_purchases_avoided_val += f_pa
            tot_residual_purchases_units += f_rem
            tot_logistics_cost += f_log
        else:
            allocations = opt_res["allocations"]
            arcs = opt_res["arcs"]

            for j_idx, dest in enumerate(destinations):
                dest_id = dest["dest_id"]
                med_id = dest["medicine_id"]
                gross_qty = dest["gross_reorder_qty"]
                unit_cost = dest["unit_cost"]

                matching_alloc_arcs = [
                    (arcs[k], allocations[k]) for k in allocations if arcs[k]["dest_idx"] == j_idx
                ]
                total_transferred = sum(t_qty for _, t_qty in matching_alloc_arcs)
                remaining_reorder = max(0, gross_qty - total_transferred)

                if matching_alloc_arcs:
                    for arc, t_qty in matching_alloc_arcs:
                        waste_avoided = round(t_qty * unit_cost, 2)
                        log_fee = round(t_qty * arc["unit_logistics"], 2)

                        tot_rescued_units += t_qty
                        tot_waste_avoided_val += waste_avoided
                        tot_purchases_avoided_val += waste_avoided
                        tot_logistics_cost += log_fee

                        recommendations.append({
                            "recommendation_id": f"REC-NET-{med_id}-{arc['source_id']}",
                            "operating_mode": MODE_NETWORK_SHARING,
                            "destination_id": "NET-SELF",
                            "destination_name": "Apex Community Pharmacy (Your Store)",
                            "source_id": arc["source_id"],
                            "source_name": arc["source_name"],
                            "medicine_id": med_id,
                            "batch_id": "PEER_LOT_MASKED",
                            "batch_expiry": arc["batch_expiry"],
                            "days_to_expiry": arc["days_to_expiry"],
                            "transfer_quantity": t_qty,
                            "original_reorder_qty": gross_qty,
                            "remaining_reorder_qty": remaining_reorder,
                            "unit_cost": unit_cost,
                            "financial_waste_avoided": waste_avoided,
                            "purchases_avoided": waste_avoided,
                            "logistics_cost": log_fee,
                            "decision": "POTENTIAL_NETWORK_MATCH",
                            "status": "POTENTIAL_MATCH",
                            "reason": (
                                f"Optimization selected {t_qty} surplus units expiring in {arc['days_to_expiry']} days "
                                f"from {arc['source_name']}. Estimated courier: ₹{log_fee:.2f}."
                            ),
                            "action_label": f"Send Stock-Sharing Request to {arc['source_id']}",
                            "residual_action": (
                                f"Reorder remaining {remaining_reorder} units from supplier"
                                if remaining_reorder > 0 else "100% Peer Rescue feasible"
                            ),
                            "optimization_status": solver_status
                        })
                    tot_residual_purchases_units += remaining_reorder
                else:
                    tot_residual_purchases_units += gross_qty
                    recommendations.append({
                        "recommendation_id": f"REC-NET-EXT-{med_id}",
                        "operating_mode": MODE_NETWORK_SHARING,
                        "destination_id": "NET-SELF",
                        "destination_name": "Apex Community Pharmacy (Your Store)",
                        "source_id": "EXTERNAL_SUPPLIER",
                        "source_name": "Authorized Distributor",
                        "medicine_id": med_id,
                        "batch_id": "NEW_ORDER",
                        "batch_expiry": "Standard Manufacturer Shelf Life",
                        "days_to_expiry": 365,
                        "transfer_quantity": 0,
                        "original_reorder_qty": gross_qty,
                        "remaining_reorder_qty": gross_qty,
                        "unit_cost": unit_cost,
                        "financial_waste_avoided": 0.0,
                        "purchases_avoided": 0.0,
                        "logistics_cost": 0.0,
                        "decision": "SUPPLIER_PURCHASE_ONLY",
                        "status": "DIRECT_REORDER",
                        "reason": f"No peer pharmacy in network holds surplus {med_id} matching expiry suitability. Standard supplier purchase recommended.",
                        "action_label": f"Reorder {gross_qty} units from supplier",
                        "residual_action": f"Purchase {gross_qty} units from supplier",
                        "optimization_status": solver_status
                    })

    else:
        # MODE_SINGLE_STORE
        b1_inv = branch_inventory[branch_inventory["branch_id"] == "BR-01"]
        for _, row in b1_inv.iterrows():
            med_id = row["medicine_id"]
            req = calculate_branch_reorder_requirement(
                current_stock=row["current_stock"],
                pending_orders=row["pending_orders"],
                daily_demand=row["daily_demand"],
                lead_time_days=int(row["lead_time_days"]),
                safety_stock=row["safety_stock"]
            )
            gross_qty = req["gross_reorder_qty"]
            tot_gross_reorder_units += gross_qty

            if req["is_reorder_needed"]:
                tot_residual_purchases_units += gross_qty
                recommendations.append({
                    "recommendation_id": f"REC-SINGLE-{med_id}",
                    "operating_mode": MODE_SINGLE_STORE,
                    "destination_id": "SINGLE_STORE",
                    "destination_name": "Single Store Location",
                    "source_id": "EXTERNAL_SUPPLIER",
                    "source_name": "Primary Wholesaler",
                    "medicine_id": med_id,
                    "batch_id": "NEW_ORDER",
                    "batch_expiry": "Standard Manufacturer Shelf Life",
                    "days_to_expiry": 365,
                    "transfer_quantity": 0,
                    "original_reorder_qty": gross_qty,
                    "remaining_reorder_qty": gross_qty,
                    "unit_cost": row["unit_cost"],
                    "financial_waste_avoided": 0.0,
                    "purchases_avoided": 0.0,
                    "logistics_cost": 0.0,
                    "decision": "SINGLE_STORE_REORDER",
                    "status": "DIRECT_REORDER",
                    "reason": f"Single-store operating model active. Stock sharing disabled. Full procurement quantity ({gross_qty} units) ordered from primary wholesaler.",
                    "action_label": f"Purchase {gross_qty} units from wholesaler",
                    "residual_action": f"Purchase {gross_qty} units from wholesaler",
                    "optimization_status": "NOT_APPLICABLE_SINGLE_STORE"
                })

    # Financial Ledger Comparison
    trad_cost = round(sum(r["original_reorder_qty"] * r["unit_cost"] for r in recommendations), 2)
    res_cost = round(sum(r["remaining_reorder_qty"] * r["unit_cost"] for r in recommendations), 2)
    rescued_val = round(tot_purchases_avoided_val, 2)
    net_procurement = round(res_cost + tot_logistics_cost, 2)
    net_savings_val = max(0.0, trad_cost - net_procurement)
    net_savings_pct = round((net_savings_val / max(1.0, trad_cost)) * 100, 1)

    comparison = {
        "traditional_purchase_units": int(tot_gross_reorder_units),
        "traditional_purchase_cost": trad_cost,
        "rescued_units": int(tot_rescued_units),
        "rescued_value_avoided": rescued_val,
        "residual_purchase_units": int(tot_residual_purchases_units),
        "residual_purchase_cost": res_cost,
        "logistics_cost": round(tot_logistics_cost, 2),
        "net_procurement_spend": net_procurement,
        "net_savings_pct": net_savings_pct
    }

    summary = {
        "potential_stock_rescued_units": int(tot_rescued_units),
        "expiry_waste_avoided_val": round(tot_waste_avoided_val, 2),
        "purchases_avoided_val": round(tot_purchases_avoided_val, 2),
        "logistics_cost_val": round(tot_logistics_cost, 2),
        "net_optimized_cost": net_procurement,
        "pending_transfer_count": sum(1 for r in recommendations if r["decision"] in ["RECOMMENDED_TRANSFER", "POTENTIAL_NETWORK_MATCH"]),
        "total_recommendations": len(recommendations),
        "solver_status": solver_status,
        "optimization_method": optimization_method
    }

    optimization_meta = {
        "solver": "scipy.optimize.milp (HiGHS)",
        "status": solver_status,
        "solver_status": solver_status,
        "method": optimization_method,
        "is_globally_optimal": solver_status == "OPTIMAL_MILP",
        "fallback_used": solver_status == "FALLBACK_GREEDY",
        "total_gross_units": int(tot_gross_reorder_units),
        "total_rescued_units": int(tot_rescued_units),
        "total_residual_units": int(tot_residual_purchases_units),
        "total_logistics_cost": round(tot_logistics_cost, 2)
    }

    return {
        "recommendations": recommendations,
        "summary": summary,
        "comparison": comparison,
        "operating_mode": operating_mode,
        "optimization_meta": optimization_meta
    }



# ---------------------------------------------------------------------------
# Authorization & State Transition Actions
# ---------------------------------------------------------------------------
def execute_confirmed_transfer(
    branch_inventory: pd.DataFrame,
    source_branch_id: str,
    dest_branch_id: str,
    medicine_id: str,
    quantity: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Authorized execution of stock transfer.
    CRITICAL SAFETY RULE: Never change inventory quantities until an authorized user confirms a transfer.
    Returns:
    - Updated branch_inventory DataFrame
    - Audit log record
    """
    df = branch_inventory.copy()
    
    # Validation checks
    src_idx = df[(df["branch_id"] == source_branch_id) & (df["medicine_id"] == medicine_id)].index
    dest_idx = df[(df["branch_id"] == dest_branch_id) & (df["medicine_id"] == medicine_id)].index

    if src_idx.empty:
        raise ValueError(f"Source branch '{source_branch_id}' does not stock medicine '{medicine_id}'.")
    if dest_idx.empty:
        raise ValueError(f"Destination branch '{dest_branch_id}' does not stock medicine '{medicine_id}'.")

    curr_src_stock = df.loc[src_idx[0], "current_stock"]
    if curr_src_stock < quantity:
        raise ValueError(f"Insufficient stock at source branch: {curr_src_stock} < {quantity}.")

    # Apply debit & credit
    df.loc[src_idx[0], "current_stock"] -= quantity
    df.loc[dest_idx[0], "current_stock"] += quantity

    audit_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_type": "INTERNAL_BRANCH_TRANSFER",
        "source_branch": source_branch_id,
        "dest_branch": dest_branch_id,
        "medicine_id": medicine_id,
        "quantity": quantity,
        "status": "COMPLETED",
        "authorized_by": "Current User (Pharmacist-in-Charge)"
    }

    return df, audit_entry
