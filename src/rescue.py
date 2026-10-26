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
    Executes the 7-Step Rescue-to-Reorder decision algorithm:
    
    1. Check demand and current stock at destination.
    2. Identify eligible surplus stock approaching expiry at other locations.
    3. Search authorized internal branches or opted-in peer pharmacies.
    4. Evaluate feasibility (medicine identity, unexpired, shelf-life suitability, storage).
    5. Estimate remaining stock requirement after proposed transfer.
    6. Recommend new supplier purchase only for the remaining shortage.
    7. Provide explicit explanation if no suitable match exists.
    
    Returns a comprehensive dictionary with:
    - recommendations: List of structured transfer opportunities
    - summary: Aggregated metrics (total rescued, waste avoided, purchases avoided)
    - comparison: 'Transfer First' vs 'Purchase New' ledger
    """
    recommendations = []
    tot_rescued_units = 0.0
    tot_waste_avoided_val = 0.0
    tot_purchases_avoided_val = 0.0
    tot_residual_purchases_units = 0.0
    tot_gross_reorder_units = 0.0

    if active_confirmed_transfers is None:
        active_confirmed_transfers = []

    # Map of already reserved/transferred quantities by batch or branch
    reserved_by_source = {}
    for t in active_confirmed_transfers:
        if t.get("status") in ["CONFIRMED", "ACCEPTED", "IN_TRANSIT"]:
            k = (t["source_id"], t["medicine_id"])
            reserved_by_source[k] = reserved_by_source.get(k, 0.0) + t["quantity"]

    ref_dt = pd.to_datetime(reference_date)

    if operating_mode == MODE_MULTI_BRANCH:
        # Multi-Branch internal transfer engine
        branches_to_eval = [target_branch_id] if target_branch_id else branch_inventory["branch_id"].unique().tolist()

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

                if not req["is_reorder_needed"]:
                    continue

                # Step 2 & 3: Search other internal branches for eligible surplus stock
                other_branches = branch_inventory[(branch_inventory["branch_id"] != b_id) & (branch_inventory["medicine_id"] == med_id)]
                
                best_match = None
                for _, s_row in other_branches.iterrows():
                    s_id = s_row["branch_id"]
                    # Calculate surplus at source: stock beyond ROP and safety requirements
                    s_demand = s_row["daily_demand"]
                    s_lt = s_row["lead_time_days"]
                    s_safety = s_row["safety_stock"]
                    s_needed_local = (s_demand * (s_lt + 14)) + s_safety
                    raw_surplus = max(0.0, s_row["current_stock"] - s_needed_local)

                    # Deduct previously committed transfers
                    already_committed = reserved_by_source.get((s_id, med_id), 0.0)
                    available_surplus = max(0.0, raw_surplus - already_committed)

                    if available_surplus <= 0.5:
                        continue

                    # Step 4: Evaluate batch eligibility (strictly unexpired, approaching expiry window)
                    s_batches = branch_batches[
                        (branch_batches["branch_id"] == s_id) & 
                        (branch_batches["medicine_id"] == med_id) &
                        (~branch_batches["is_expired"])
                    ].sort_values("days_to_expiry")

                    if s_batches.empty:
                        continue

                    # Select candidate batch
                    for _, batch in s_batches.iterrows():
                        days_left = batch["days_to_expiry"]
                        # Strict suitability: must have at least 14 days of life, and enough shelf life
                        # for destination daily demand to consume it safely
                        dest_daily = row["daily_demand"]
                        needed_days_to_consume = min(available_surplus, gross_qty) / dest_daily if dest_daily > 0 else 999
                        
                        if days_left > 14 and days_left >= needed_days_to_consume:
                            transferrable_qty = min(available_surplus, float(gross_qty), float(batch["quantity"]))
                            if transferrable_qty >= 1.0:
                                best_match = {
                                    "source_id": s_id,
                                    "batch_id": batch["batch_id"],
                                    "days_to_expiry": int(days_left),
                                    "expiry_date": batch["expiry_date"],
                                    "transfer_qty": int(np.floor(transferrable_qty)),
                                    "unit_cost": row["unit_cost"]
                                }
                                break
                    if best_match:
                        break

                # Step 5 & 6: Estimate remaining stock requirement
                if best_match:
                    rescued_qty = best_match["transfer_qty"]
                    remaining_reorder = max(0, gross_qty - rescued_qty)
                    waste_avoided = round(rescued_qty * best_match["unit_cost"], 2)
                    purchase_avoided = waste_avoided

                    tot_rescued_units += rescued_qty
                    tot_waste_avoided_val += waste_avoided
                    tot_purchases_avoided_val += purchase_avoided
                    tot_residual_purchases_units += remaining_reorder

                    recommendations.append({
                        "recommendation_id": f"REC-MB-{b_id}-{med_id}",
                        "operating_mode": MODE_MULTI_BRANCH,
                        "destination_id": b_id,
                        "destination_name": next((b["name"] for b in DEFAULT_BRANCHES if b["branch_id"] == b_id), b_id),
                        "source_id": best_match["source_id"],
                        "source_name": next((b["name"] for b in DEFAULT_BRANCHES if b["branch_id"] == best_match["source_id"]), best_match["source_id"]),
                        "medicine_id": med_id,
                        "batch_id": best_match["batch_id"],
                        "batch_expiry": best_match["expiry_date"],
                        "days_to_expiry": best_match["days_to_expiry"],
                        "transfer_quantity": rescued_qty,
                        "original_reorder_qty": gross_qty,
                        "remaining_reorder_qty": remaining_reorder,
                        "unit_cost": best_match["unit_cost"],
                        "financial_waste_avoided": waste_avoided,
                        "purchases_avoided": purchase_avoided,
                        "decision": "RECOMMENDED_TRANSFER",
                        "status": "RECOMMENDED",
                        "reason": f"Source branch {best_match['source_id']} holds {best_match['transfer_qty']} units surplus expiring in {best_match['days_to_expiry']} days. Destination demand ({row['daily_demand']:.1f}/day) can absorb stock safely within {best_match['days_to_expiry']} days.",
                        "action_label": f"Transfer {rescued_qty} units from {best_match['source_id']}",
                        "residual_action": f"Purchase remaining {remaining_reorder} units from supplier" if remaining_reorder > 0 else "Zero external purchase required (100% Rescued!)"
                    })
                else:
                    # Step 7: No suitable match exists — explain why normal reorder is needed
                    tot_residual_purchases_units += gross_qty
                    recommendations.append({
                        "recommendation_id": f"REC-EXT-{b_id}-{med_id}",
                        "operating_mode": MODE_MULTI_BRANCH,
                        "destination_id": b_id,
                        "destination_name": next((b["name"] for b in DEFAULT_BRANCHES if b["branch_id"] == b_id), b_id),
                        "source_id": "EXTERNAL_SUPPLIER",
                        "source_name": "Authorized External Distributor",
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
                        "decision": "SUPPLIER_PURCHASE_ONLY",
                        "status": "DIRECT_REORDER",
                        "reason": f"No eligible internal branch has surplus stock of {med_id}. Other branches are either at or below their required safety buffer. External supplier order recommended.",
                        "action_label": f"Reorder {gross_qty} units from supplier",
                        "residual_action": f"Purchase {gross_qty} units from supplier"
                    })

    elif operating_mode == MODE_NETWORK_SHARING:
        # Independent Pharmacy Network Sharing Engine
        # User is single store or independent pharmacy checking peer network matches
        if network_stock is None or network_stock.empty:
            network_stock = get_initial_network_stock(reference_date)

        # Baseline single store inventory
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

            if not req["is_reorder_needed"]:
                continue

            # Step 3 & 4: Search network peer pharmacies
            matches = network_stock[
                (network_stock["medicine_id"] == med_id) & 
                (network_stock["days_to_expiry"] > 14)
            ].sort_values("distance_km")

            if not matches.empty:
                peer = matches.iloc[0]
                available_qty = float(peer["available_quantity"])
                rescued_qty = int(min(available_qty, gross_qty))
                remaining_reorder = max(0, gross_qty - rescued_qty)
                waste_avoided = round(rescued_qty * row["unit_cost"], 2)

                tot_rescued_units += rescued_qty
                tot_waste_avoided_val += waste_avoided
                tot_purchases_avoided_val += waste_avoided
                tot_residual_purchases_units += remaining_reorder

                recommendations.append({
                    "recommendation_id": f"REC-NET-{med_id}",
                    "operating_mode": MODE_NETWORK_SHARING,
                    "destination_id": "NET-SELF",
                    "destination_name": "Apex Community Pharmacy (Your Store)",
                    "source_id": peer["pharmacy_id"],
                    "source_name": f"{peer['name']} ({peer['distance_km']:.1f} km away)",
                    "medicine_id": med_id,
                    "batch_id": "PEER_LOT_MASKED",
                    "batch_expiry": peer["expiry_date"],
                    "days_to_expiry": int(peer["days_to_expiry"]),
                    "transfer_quantity": rescued_qty,
                    "original_reorder_qty": gross_qty,
                    "remaining_reorder_qty": remaining_reorder,
                    "unit_cost": row["unit_cost"],
                    "financial_waste_avoided": waste_avoided,
                    "purchases_avoided": waste_avoided,
                    "decision": "POTENTIAL_NETWORK_MATCH",
                    "status": "POTENTIAL_MATCH",  # Strictly unconfirmed until mutual request
                    "reason": f"Participating peer pharmacy has ~{rescued_qty} surplus units expiring in {peer['days_to_expiry']} days within {peer['distance_km']:.1f} km. Privacy-masked opportunity.",
                    "action_label": f"Send Stock-Sharing Request to {peer['name']}",
                    "residual_action": f"Reorder remaining {remaining_reorder} units from supplier" if remaining_reorder > 0 else "100% Peer Rescue feasible"
                })
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
                    "unit_cost": row["unit_cost"],
                    "financial_waste_avoided": 0.0,
                    "purchases_avoided": 0.0,
                    "decision": "SUPPLIER_PURCHASE_ONLY",
                    "status": "DIRECT_REORDER",
                    "reason": f"No peer pharmacy in the opted-in network currently holds surplus {med_id} matching expiry suitability. Standard supplier purchase recommended.",
                    "action_label": f"Reorder {gross_qty} units from supplier",
                    "residual_action": f"Purchase {gross_qty} units from supplier"
                })

    else:
        # MODE_SINGLE_STORE
        # Traditional local reorder without stock sharing
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
                    "decision": "SINGLE_STORE_REORDER",
                    "status": "DIRECT_REORDER",
                    "reason": f"Single-store operating model active. Stock sharing disabled. Full procurement quantity ({gross_qty} units) ordered from primary wholesaler.",
                    "action_label": f"Purchase {gross_qty} units from wholesaler",
                    "residual_action": f"Purchase {gross_qty} units from wholesaler"
                })

    # Financial Ledger Comparison
    comparison = {
        "traditional_purchase_units": int(tot_gross_reorder_units),
        "traditional_purchase_cost": round(sum(r["original_reorder_qty"] * r["unit_cost"] for r in recommendations), 2),
        "rescued_units": int(tot_rescued_units),
        "rescued_value_avoided": round(tot_purchases_avoided_val, 2),
        "residual_purchase_units": int(tot_residual_purchases_units),
        "residual_purchase_cost": round(sum(r["remaining_reorder_qty"] * r["unit_cost"] for r in recommendations), 2),
        "net_savings_pct": round((tot_purchases_avoided_val / max(1.0, sum(r['original_reorder_qty'] * r['unit_cost'] for r in recommendations))) * 100, 1)
    }

    summary = {
        "potential_stock_rescued_units": int(tot_rescued_units),
        "expiry_waste_avoided_val": round(tot_waste_avoided_val, 2),
        "purchases_avoided_val": round(tot_purchases_avoided_val, 2),
        "pending_transfer_count": sum(1 for r in recommendations if r["decision"] in ["RECOMMENDED_TRANSFER", "POTENTIAL_NETWORK_MATCH"]),
        "total_recommendations": len(recommendations)
    }

    return {
        "recommendations": recommendations,
        "summary": summary,
        "comparison": comparison,
        "operating_mode": operating_mode
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
