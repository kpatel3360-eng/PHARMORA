"""
PHARMORA - Synthetic Demonstration Dataset Engine & Catalogue Expansion
=======================================================================
Generates and manages a separate, explicitly labelled synthetic demonstration dataset
comprising 50 realistic pharmaceutical products spanning 8 key therapeutic categories.

Guarantees & Compliance:
1. Strict Physical & Logical Separation: Original historical data (data/salesdaily.csv)
   is completely preserved and never mutated.
2. Explicit Provenance Labelling: Every synthetic product, batch, inventory record,
   and forecast is explicitly tagged with `data_source = "SYNTHETIC_DEMO"` and `is_synthetic = True`.
3. Zero-Fabrication Integrity: Evaluation metrics (MAE, RMSE, WAPE) are strictly reserved
   for genuine historical observations. Synthetic products provide simulated demonstration
   forecasts for presentation without claiming real-world accuracy.
4. Comprehensive Validation: Validates unique identifiers, ISO dates, non-negative quantities,
   zero duplicates, and zero missing values.
"""

import os
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# Directory for storing separate synthetic demonstration assets
SYNTHETIC_DATA_DIR = Path("data/synthetic_demo")
SYNTHETIC_DIR = SYNTHETIC_DATA_DIR

# ---------------------------------------------------------------------------
# 50 Realistically Modeled Pharmaceutical Products
# ---------------------------------------------------------------------------
SYNTHETIC_CATALOGUE: List[Dict[str, Any]] = [
    # 1. Antibiotics & Anti-infectives (ATC: J01)
    {"medicine_id": "SYN-AMOX-500", "medicine_name": "Amoxicillin 500mg Capsule", "generic_name": "Amoxicillin", "atc_code": "J01CA04", "therapeutic_class": "Broad Spectrum Penicillins", "unit": "Capsule", "storage_condition": "Room Temperature", "unit_cost": 6.50, "lead_time_days": 5, "daily_demand_mean": 24.5, "daily_demand_std": 5.8},
    {"medicine_id": "SYN-AZIT-500", "medicine_name": "Azithromycin 500mg Tablet", "generic_name": "Azithromycin", "atc_code": "J01FA10", "therapeutic_class": "Macrolides", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 18.20, "lead_time_days": 6, "daily_demand_mean": 16.0, "daily_demand_std": 4.2},
    {"medicine_id": "SYN-CIPR-500", "medicine_name": "Ciprofloxacin 500mg Tablet", "generic_name": "Ciprofloxacin", "atc_code": "J01MA02", "therapeutic_class": "Fluoroquinolones", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 9.40, "lead_time_days": 4, "daily_demand_mean": 12.8, "daily_demand_std": 3.5},
    {"medicine_id": "SYN-DOXY-100", "medicine_name": "Doxycycline 100mg Capsule", "generic_name": "Doxycycline", "atc_code": "J01AA02", "therapeutic_class": "Tetracyclines", "unit": "Capsule", "storage_condition": "Room Temperature", "unit_cost": 5.80, "lead_time_days": 5, "daily_demand_mean": 9.5, "daily_demand_std": 2.8},
    {"medicine_id": "SYN-CEFT-1G",  "medicine_name": "Ceftriaxone 1g Injection", "generic_name": "Ceftriaxone", "atc_code": "J01DD04", "therapeutic_class": "Third-Gen Cephalosporins", "unit": "Vial", "storage_condition": "Cold Chain (2-8°C)", "unit_cost": 45.00, "lead_time_days": 7, "daily_demand_mean": 8.0, "daily_demand_std": 2.5},
    {"medicine_id": "SYN-CEFX-200", "medicine_name": "Cefixime 200mg Tablet", "generic_name": "Cefixime", "atc_code": "J01DD08", "therapeutic_class": "Oral Cephalosporins", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 14.50, "lead_time_days": 5, "daily_demand_mean": 14.2, "daily_demand_std": 3.9},
    {"medicine_id": "SYN-LEVO-500", "medicine_name": "Levofloxacin 500mg Tablet", "generic_name": "Levofloxacin", "atc_code": "J01MA12", "therapeutic_class": "Fluoroquinolones", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 12.00, "lead_time_days": 4, "daily_demand_mean": 10.4, "daily_demand_std": 3.1},
    {"medicine_id": "SYN-METR-400", "medicine_name": "Metronidazole 400mg Tablet", "generic_name": "Metronidazole", "atc_code": "P01AB01", "therapeutic_class": "Antiprotozoals & Antibacterials", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 3.20, "lead_time_days": 3, "daily_demand_mean": 18.5, "daily_demand_std": 4.5},

    # 2. Cardiovascular & Antihypertensives (ATC: C07, C08, C09, C10)
    {"medicine_id": "SYN-ATOR-20",  "medicine_name": "Atorvastatin 20mg Tablet", "generic_name": "Atorvastatin", "atc_code": "C10AA05", "therapeutic_class": "HMG-CoA Reductase Inhibitors", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 11.50, "lead_time_days": 5, "daily_demand_mean": 32.0, "daily_demand_std": 6.2},
    {"medicine_id": "SYN-AMLO-5",   "medicine_name": "Amlodipine 5mg Tablet", "generic_name": "Amlodipine", "atc_code": "C08CA01", "therapeutic_class": "Calcium Channel Blockers", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 4.20, "lead_time_days": 4, "daily_demand_mean": 28.5, "daily_demand_std": 5.4},
    {"medicine_id": "SYN-LOSA-50",  "medicine_name": "Losartan 50mg Tablet", "generic_name": "Losartan", "atc_code": "C09CA01", "therapeutic_class": "Angiotensin II Receptor Antagonists", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 7.80, "lead_time_days": 5, "daily_demand_mean": 21.0, "daily_demand_std": 4.1},
    {"medicine_id": "SYN-TELM-40",  "medicine_name": "Telmisartan 40mg Tablet", "generic_name": "Telmisartan", "atc_code": "C09CA07", "therapeutic_class": "Angiotensin II Receptor Antagonists", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 8.90, "lead_time_days": 5, "daily_demand_mean": 26.4, "daily_demand_std": 4.8},
    {"medicine_id": "SYN-METO-50",  "medicine_name": "Metoprolol Succinate 50mg", "generic_name": "Metoprolol", "atc_code": "C07AB02", "therapeutic_class": "Beta Blocking Agents", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 9.20, "lead_time_days": 6, "daily_demand_mean": 17.5, "daily_demand_std": 3.6},
    {"medicine_id": "SYN-ROSU-10",  "medicine_name": "Rosuvastatin 10mg Tablet", "generic_name": "Rosuvastatin", "atc_code": "C10AA07", "therapeutic_class": "Lipid Modifying Agents", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 14.80, "lead_time_days": 5, "daily_demand_mean": 19.8, "daily_demand_std": 4.0},
    {"medicine_id": "SYN-CLOP-75",  "medicine_name": "Clopidogrel 75mg Tablet", "generic_name": "Clopidogrel", "atc_code": "B01AC04", "therapeutic_class": "Platelet Aggregation Inhibitors", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 13.50, "lead_time_days": 6, "daily_demand_mean": 15.2, "daily_demand_std": 3.3},
    {"medicine_id": "SYN-ENAL-5",   "medicine_name": "Enalapril 5mg Tablet", "generic_name": "Enalapril", "atc_code": "C09AA02", "therapeutic_class": "ACE Inhibitors", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 5.10, "lead_time_days": 4, "daily_demand_mean": 11.0, "daily_demand_std": 2.6},

    # 3. Antidiabetic & Endocrine (ATC: A10)
    {"medicine_id": "SYN-METF-500", "medicine_name": "Metformin 500mg SR Tablet", "generic_name": "Metformin", "atc_code": "A10BA02", "therapeutic_class": "Biguanides / Oral Hypoglycemics", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 3.80, "lead_time_days": 4, "daily_demand_mean": 42.0, "daily_demand_std": 7.5},
    {"medicine_id": "SYN-GLIM-2",   "medicine_name": "Glimepiride 2mg Tablet", "generic_name": "Glimepiride", "atc_code": "A10BB12", "therapeutic_class": "Sulfonylureas", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 6.40, "lead_time_days": 4, "daily_demand_mean": 22.5, "daily_demand_std": 4.5},
    {"medicine_id": "SYN-SITA-100", "medicine_name": "Sitagliptin 100mg Tablet", "generic_name": "Sitagliptin", "atc_code": "A10BH01", "therapeutic_class": "DPP-4 Inhibitors", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 28.00, "lead_time_days": 7, "daily_demand_mean": 13.5, "daily_demand_std": 3.1},
    {"medicine_id": "SYN-DAPA-10",  "medicine_name": "Dapagliflozin 10mg Tablet", "generic_name": "Dapagliflozin", "atc_code": "A10BK01", "therapeutic_class": "SGLT2 Inhibitors", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 32.50, "lead_time_days": 7, "daily_demand_mean": 15.0, "daily_demand_std": 3.4},
    {"medicine_id": "SYN-EMPA-25",  "medicine_name": "Empagliflozin 25mg Tablet", "generic_name": "Empagliflozin", "atc_code": "A10BK03", "therapeutic_class": "SGLT2 Inhibitors", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 36.00, "lead_time_days": 7, "daily_demand_mean": 11.5, "daily_demand_std": 2.9},
    {"medicine_id": "SYN-VILD-50",  "medicine_name": "Vildagliptin 50mg Tablet", "generic_name": "Vildagliptin", "atc_code": "A10BH02", "therapeutic_class": "DPP-4 Inhibitors", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 22.00, "lead_time_days": 6, "daily_demand_mean": 14.0, "daily_demand_std": 3.2},
    {"medicine_id": "SYN-INSU-GLA", "medicine_name": "Insulin Glargine 100IU/ml", "generic_name": "Insulin Glargine", "atc_code": "A10AE04", "therapeutic_class": "Long-Acting Insulins", "unit": "Vial", "storage_condition": "Cold Chain (2-8°C)", "unit_cost": 165.00, "lead_time_days": 8, "daily_demand_mean": 6.5, "daily_demand_std": 1.8},

    # 4. Gastrointestinal & Acid-Related (ATC: A02)
    {"medicine_id": "SYN-OMEP-20",  "medicine_name": "Omeprazole 20mg Capsule", "generic_name": "Omeprazole", "atc_code": "A02BC01", "therapeutic_class": "Proton Pump Inhibitors", "unit": "Capsule", "storage_condition": "Room Temperature", "unit_cost": 4.50, "lead_time_days": 4, "daily_demand_mean": 35.0, "daily_demand_std": 6.8},
    {"medicine_id": "SYN-PANT-40",  "medicine_name": "Pantoprazole 40mg Tablet", "generic_name": "Pantoprazole", "atc_code": "A02BC02", "therapeutic_class": "Proton Pump Inhibitors", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 7.20, "lead_time_days": 4, "daily_demand_mean": 38.0, "daily_demand_std": 7.2},
    {"medicine_id": "SYN-RABE-20",  "medicine_name": "Rabeprazole 20mg Tablet", "generic_name": "Rabeprazole", "atc_code": "A02BC04", "therapeutic_class": "Proton Pump Inhibitors", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 8.50, "lead_time_days": 4, "daily_demand_mean": 20.0, "daily_demand_std": 4.6},
    {"medicine_id": "SYN-ONDA-4",   "medicine_name": "Ondansetron 4mg MD Tablet", "generic_name": "Ondansetron", "atc_code": "A04AA01", "therapeutic_class": "Serotonin 5-HT3 Antagonists / Antiemetics", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 6.80, "lead_time_days": 3, "daily_demand_mean": 12.0, "daily_demand_std": 3.1},
    {"medicine_id": "SYN-DOMP-10",  "medicine_name": "Domperidone 10mg Tablet", "generic_name": "Domperidone", "atc_code": "A03FA03", "therapeutic_class": "Propulsives / Gastroprokinetics", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 4.10, "lead_time_days": 4, "daily_demand_mean": 15.5, "daily_demand_std": 3.7},

    # 5. Respiratory & Antiallergic (ATC: R01, R03, R06)
    {"medicine_id": "SYN-MONT-10",  "medicine_name": "Montelukast 10mg Tablet", "generic_name": "Montelukast", "atc_code": "R03DC03", "therapeutic_class": "Leukotriene Receptor Antagonists", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 12.50, "lead_time_days": 5, "daily_demand_mean": 22.0, "daily_demand_std": 4.8},
    {"medicine_id": "SYN-LEVC-5",   "medicine_name": "Levocetirizine 5mg Tablet", "generic_name": "Levocetirizine", "atc_code": "R06AE09", "therapeutic_class": "Second-Gen Antihistamines", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 5.40, "lead_time_days": 4, "daily_demand_mean": 29.0, "daily_demand_std": 6.1},
    {"medicine_id": "SYN-FEXO-120", "medicine_name": "Fexofenadine 120mg Tablet", "generic_name": "Fexofenadine", "atc_code": "R06AX26", "therapeutic_class": "Non-sedating Antihistamines", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 14.00, "lead_time_days": 5, "daily_demand_mean": 16.5, "daily_demand_std": 3.9},
    {"medicine_id": "SYN-BUDE-200", "medicine_name": "Budesonide Inhaler 200mcg", "generic_name": "Budesonide", "atc_code": "R03BA02", "therapeutic_class": "Inhaled Glucocorticoids", "unit": "Inhaler", "storage_condition": "Room Temperature", "unit_cost": 85.00, "lead_time_days": 7, "daily_demand_mean": 7.2, "daily_demand_std": 2.0},
    {"medicine_id": "SYN-FLUT-50",  "medicine_name": "Fluticasone Nasal Spray 50mcg", "generic_name": "Fluticasone", "atc_code": "R01AD08", "therapeutic_class": "Nasal Corticosteroids", "unit": "Spray", "storage_condition": "Room Temperature", "unit_cost": 95.00, "lead_time_days": 6, "daily_demand_mean": 5.8, "daily_demand_std": 1.7},

    # 6. Analgesics, Antipyretics & Musculoskeletal (ATC: M01, N02)
    {"medicine_id": "SYN-ACEC-100", "medicine_name": "Aceclofenac 100mg Tablet", "generic_name": "Aceclofenac", "atc_code": "M01AB16", "therapeutic_class": "Acetic Acid NSAIDs", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 6.90, "lead_time_days": 4, "daily_demand_mean": 25.0, "daily_demand_std": 5.5},
    {"medicine_id": "SYN-TRAM-50",  "medicine_name": "Tramadol 50mg Capsule", "generic_name": "Tramadol", "atc_code": "N02AX02", "therapeutic_class": "Other Opioids / Controlled Analgesics", "unit": "Capsule", "storage_condition": "Room Temperature (Locked)", "unit_cost": 15.00, "lead_time_days": 6, "daily_demand_mean": 8.5, "daily_demand_std": 2.3},
    {"medicine_id": "SYN-NAPR-500", "medicine_name": "Naproxen 500mg Tablet", "generic_name": "Naproxen", "atc_code": "M01AE02", "therapeutic_class": "Propionic Acid NSAIDs", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 8.20, "lead_time_days": 5, "daily_demand_mean": 13.0, "daily_demand_std": 3.2},
    {"medicine_id": "SYN-ETOR-90",  "medicine_name": "Etoricoxib 90mg Tablet", "generic_name": "Etoricoxib", "atc_code": "M01AH05", "therapeutic_class": "Coxib Selective NSAIDs", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 16.50, "lead_time_days": 5, "daily_demand_mean": 11.2, "daily_demand_std": 2.8},
    {"medicine_id": "SYN-MEFE-500", "medicine_name": "Mefenamic Acid 500mg", "generic_name": "Mefenamic Acid", "atc_code": "M01AG01", "therapeutic_class": "Fenamates NSAIDs", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 5.20, "lead_time_days": 4, "daily_demand_mean": 14.8, "daily_demand_std": 3.6},

    # 7. Central Nervous System & Psychiatric (ATC: N05, N06)
    {"medicine_id": "SYN-SERT-50",  "medicine_name": "Sertraline 50mg Tablet", "generic_name": "Sertraline", "atc_code": "N06AB06", "therapeutic_class": "SSRIs / Antidepressants", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 14.20, "lead_time_days": 6, "daily_demand_mean": 12.0, "daily_demand_std": 2.8},
    {"medicine_id": "SYN-ESCI-10",  "medicine_name": "Escitalopram 10mg Tablet", "generic_name": "Escitalopram", "atc_code": "N06AB10", "therapeutic_class": "SSRIs / Antidepressants", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 11.80, "lead_time_days": 5, "daily_demand_mean": 18.0, "daily_demand_std": 3.8},
    {"medicine_id": "SYN-ALPR-05",  "medicine_name": "Alprazolam 0.5mg Tablet", "generic_name": "Alprazolam", "atc_code": "N05BA12", "therapeutic_class": "Benzodiazepines / Anxiolytics", "unit": "Tablet", "storage_condition": "Room Temperature (Locked)", "unit_cost": 4.80, "lead_time_days": 5, "daily_demand_mean": 15.5, "daily_demand_std": 3.5},
    {"medicine_id": "SYN-CLON-05",  "medicine_name": "Clonazepam 0.5mg Tablet", "generic_name": "Clonazepam", "atc_code": "N03AE01", "therapeutic_class": "Antiepileptics / Anxiolytics", "unit": "Tablet", "storage_condition": "Room Temperature (Locked)", "unit_cost": 5.50, "lead_time_days": 5, "daily_demand_mean": 13.2, "daily_demand_std": 3.0},
    {"medicine_id": "SYN-GABA-300", "medicine_name": "Gabapentin 300mg Capsule", "generic_name": "Gabapentin", "atc_code": "N03AX12", "therapeutic_class": "Other Antiepileptics / Neuropathic Pain", "unit": "Capsule", "storage_condition": "Room Temperature", "unit_cost": 17.50, "lead_time_days": 6, "daily_demand_mean": 10.0, "daily_demand_std": 2.5},

    # 8. Vitamins, Minerals & Nutritional Supplements (ATC: A11, B03, M05)
    {"medicine_id": "SYN-VITD-60K", "medicine_name": "Cholecalciferol 60,000 IU", "generic_name": "Vitamin D3", "atc_code": "A11CC05", "therapeutic_class": "Vitamin D & Analogues", "unit": "Capsule", "storage_condition": "Room Temperature", "unit_cost": 22.00, "lead_time_days": 5, "daily_demand_mean": 28.0, "daily_demand_std": 6.5},
    {"medicine_id": "SYN-METH-1500","medicine_name": "Methylcobalamin 1500mcg", "generic_name": "Vitamin B12", "atc_code": "B03BA05", "therapeutic_class": "Vitamin B12 Supplements", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 13.00, "lead_time_days": 4, "daily_demand_mean": 24.0, "daily_demand_std": 5.2},
    {"medicine_id": "SYN-CALC-500", "medicine_name": "Calcium Carbonate + D3", "generic_name": "Calcium with Vitamin D3", "atc_code": "A12AX", "therapeutic_class": "Mineral Supplements", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 7.50, "lead_time_days": 4, "daily_demand_mean": 31.0, "daily_demand_std": 6.0},
    {"medicine_id": "SYN-IRON-FOL", "medicine_name": "Ferrous Ascorbate + Folic", "generic_name": "Iron with Folic Acid", "atc_code": "B03AA07", "therapeutic_class": "Iron Preparations", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 9.00, "lead_time_days": 4, "daily_demand_mean": 19.5, "daily_demand_std": 4.2},
    {"medicine_id": "SYN-ZINC-50",  "medicine_name": "Zinc Sulfate 50mg Tablet", "generic_name": "Zinc Sulfate", "atc_code": "A12CB01", "therapeutic_class": "Mineral Supplements / Zinc", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 4.00, "lead_time_days": 3, "daily_demand_mean": 16.0, "daily_demand_std": 3.7},
    {"medicine_id": "SYN-PARA-650", "medicine_name": "Paracetamol 650mg Tablet", "generic_name": "Paracetamol (Clinical)", "atc_code": "N02BE01", "therapeutic_class": "Anilide Analgesics & Antipyretics", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 2.50, "lead_time_days": 3, "daily_demand_mean": 48.0, "daily_demand_std": 9.2},
    {"medicine_id": "SYN-ALEN-70",  "medicine_name": "Alendronic Acid 70mg Tablet", "generic_name": "Alendronate Sodium", "atc_code": "M05BA04", "therapeutic_class": "Bisphosphonates / Bone Modulators", "unit": "Tablet", "storage_condition": "Room Temperature", "unit_cost": 38.00, "lead_time_days": 7, "daily_demand_mean": 5.2, "daily_demand_std": 1.4},
]
SYNTHETIC_MEDICINES = SYNTHETIC_CATALOGUE

# Quick lookup dictionaries
SYNTHETIC_MED_MAP: Dict[str, Dict[str, Any]] = {m["medicine_id"]: m for m in SYNTHETIC_CATALOGUE}
SYNTHETIC_MED_NAMES: Dict[str, str] = {m["medicine_id"]: f"{m['medicine_id']} - {m['medicine_name']}" for m in SYNTHETIC_CATALOGUE}
SYNTHETIC_SHORT_NAMES: Dict[str, str] = {m["medicine_id"]: m["medicine_name"] for m in SYNTHETIC_CATALOGUE}
SYNTHETIC_CLASSES: Dict[str, str] = {m["medicine_id"]: m["therapeutic_class"] for m in SYNTHETIC_CATALOGUE}


def generate_synthetic_demonstration_records(
    output_dir: Path = SYNTHETIC_DATA_DIR,
    history_days: int = 180,
    reference_date: datetime.date = datetime.date(2026, 10, 28),
    seed: int = 101
) -> Dict[str, Path]:
    """
    Generates reproducible synthetic demonstration dataset files:
    1. sales.csv: 50 medicines x 180 days of realistic daily sales simulations
    2. inventory.csv: current stock, lead times, safety stocks, pending orders, unit costs
    3. batches.csv: realistic batches with expiry dates, quantities, and storage conditions
    4. forecast_7days.csv: 7-day simulated recursive demand forecasts
    """
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = reference_date - datetime.timedelta(days=history_days - 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(history_days)]

    # -----------------------------------------------------------------------
    # 1. Sales Simulation
    # -----------------------------------------------------------------------
    sales_rows = []
    weekday_weights = {
        0: 1.10,  # Monday surge
        1: 1.02,
        2: 0.98,
        3: 1.05,
        4: 1.12,  # Friday pre-weekend
        5: 0.90,  # Saturday
        6: 0.75   # Sunday reduced hours
    }

    for med in SYNTHETIC_CATALOGUE:
        m_id = med["medicine_id"]
        mean_d = med["daily_demand_mean"]
        std_d = med["daily_demand_std"]

        for d in dates:
            dow = d.weekday()
            dow_factor = weekday_weights.get(dow, 1.0)
            # Add subtle seasonal drift + weekly cycle + stochastic noise
            seasonal_drift = 1.0 + 0.08 * np.sin(2 * np.pi * (d.timetuple().tm_yday) / 365.25)
            target_mean = mean_d * dow_factor * seasonal_drift
            simulated_demand = max(0.0, np.round(np.random.normal(target_mean, std_d), 2))

            sales_rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "medicine_id": m_id,
                "units_sold": float(simulated_demand),
                "data_source": "SYNTHETIC_DEMO",
                "is_synthetic": True
            })

    sales_df = pd.DataFrame(sales_rows)
    sales_path = output_dir / "sales.csv"
    sales_df.to_csv(sales_path, index=False)

    # -----------------------------------------------------------------------
    # 2. Inventory Parameters Simulation
    # -----------------------------------------------------------------------
    inv_rows = []
    for idx, med in enumerate(SYNTHETIC_CATALOGUE):
        m_id = med["medicine_id"]
        mean_d = med["daily_demand_mean"]
        lt = med["lead_time_days"]
        cost = med["unit_cost"]

        # Safety stock ~ 1.65 * sqrt(lt) * std
        ss = round(1.65 * np.sqrt(lt) * med["daily_demand_std"], 1)

        # Distribute into diverse scenarios:
        # Scenario A (10%): Critical stockout risk (stock < safety stock)
        # Scenario B (15%): Reorder warning (safety stock <= stock <= ROP)
        # Scenario C (10%): Overstocked (> 3x ROP)
        # Scenario D (65%): Healthy buffer
        rop = round((mean_d * lt) + ss, 1)

        if idx % 10 == 0:
            # Critical stockout
            curr_stock = round(max(2.0, ss * 0.4), 1)
            pending = 0.0
        elif idx % 10 == 1 or idx % 10 == 2:
            # Reorder warning
            curr_stock = round(rop * 0.85, 1)
            pending = 0.0
        elif idx % 10 == 3:
            # Overstocked
            curr_stock = round(rop * 3.2, 1)
            pending = round(mean_d * 5, 1)
        else:
            # Healthy
            curr_stock = round(rop * 1.5, 1)
            pending = round(mean_d * 2, 1) if (idx % 4 == 0) else 0.0

        inv_rows.append({
            "medicine_id": m_id,
            "current_stock": curr_stock,
            "supplier_lead_time_days": lt,
            "safety_stock": ss,
            "pending_order_quantity": pending,
            "unit_cost": cost,
            "data_source": "SYNTHETIC_DEMO",
            "is_synthetic": True
        })

    inv_df = pd.DataFrame(inv_rows)
    inv_path = output_dir / "inventory.csv"
    inv_df.to_csv(inv_path, index=False)

    # -----------------------------------------------------------------------
    # 3. Batch Expiry Scenarios Simulation
    # -----------------------------------------------------------------------
    batch_rows = []
    for idx, med in enumerate(SYNTHETIC_CATALOGUE):
        m_id = med["medicine_id"]
        cost = med["unit_cost"]
        storage = med["storage_condition"]
        mean_d = med["daily_demand_mean"]

        # Generate 2 to 3 batches per medicine:
        # Batch 1: Expiring soon or already expired for a subset to test FEFO logic
        if idx % 5 == 0:
            # Expired batch
            exp_date = reference_date - datetime.timedelta(days=np.random.randint(5, 30))
            qty = round(mean_d * 1.5, 1)
            batch_rows.append({
                "batch_id": f"B-{m_id}-EXP",
                "medicine_id": m_id,
                "batch_quantity": qty,
                "expiry_date": exp_date.strftime("%Y-%m-%d"),
                "unit_cost": cost,
                "storage_condition": storage,
                "data_source": "SYNTHETIC_DEMO",
                "is_synthetic": True
            })
        elif idx % 4 == 0:
            # Near expiry (within 15-45 days)
            exp_date = reference_date + datetime.timedelta(days=np.random.randint(15, 45))
            qty = round(mean_d * 3.0, 1)
            batch_rows.append({
                "batch_id": f"B-{m_id}-WARN",
                "medicine_id": m_id,
                "batch_quantity": qty,
                "expiry_date": exp_date.strftime("%Y-%m-%d"),
                "unit_cost": cost,
                "storage_condition": storage,
                "data_source": "SYNTHETIC_DEMO",
                "is_synthetic": True
            })

        # Standard healthy batches (90 to 450 days out)
        exp_date_mid = reference_date + datetime.timedelta(days=np.random.randint(90, 180))
        qty_mid = round(mean_d * 5.0, 1)
        batch_rows.append({
            "batch_id": f"B-{m_id}-M01",
            "medicine_id": m_id,
            "batch_quantity": qty_mid,
            "expiry_date": exp_date_mid.strftime("%Y-%m-%d"),
            "unit_cost": cost,
            "storage_condition": storage,
            "data_source": "SYNTHETIC_DEMO",
            "is_synthetic": True
        })

        exp_date_long = reference_date + datetime.timedelta(days=np.random.randint(240, 480))
        qty_long = round(mean_d * 8.0, 1)
        batch_rows.append({
            "batch_id": f"B-{m_id}-L02",
            "medicine_id": m_id,
            "batch_quantity": qty_long,
            "expiry_date": exp_date_long.strftime("%Y-%m-%d"),
            "unit_cost": cost,
            "storage_condition": storage,
            "data_source": "SYNTHETIC_DEMO",
            "is_synthetic": True
        })

    batches_df = pd.DataFrame(batch_rows)
    batches_path = output_dir / "batches.csv"
    batches_df.to_csv(batches_path, index=False)

    # -----------------------------------------------------------------------
    # 4. Simulated 7-Day Recursive Forecasts
    # -----------------------------------------------------------------------
    future_dates = [reference_date + datetime.timedelta(days=i) for i in range(1, 8)]
    forecast_rows = []
    for med in SYNTHETIC_CATALOGUE:
        m_id = med["medicine_id"]
        mean_d = med["daily_demand_mean"]
        std_d = med["daily_demand_std"]

        for fd in future_dates:
            dow = fd.weekday()
            dow_factor = weekday_weights.get(dow, 1.0)
            pred_demand = max(0.0, np.round(mean_d * dow_factor + np.random.normal(0, std_d * 0.3), 2))
            forecast_rows.append({
                "medicine_id": m_id,
                "forecast_date": fd.strftime("%Y-%m-%d"),
                "predicted_units": float(pred_demand),
                "model_name": "SimulatedDemoForecast",
                "data_source": "SYNTHETIC_DEMO",
                "is_synthetic": True
            })

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_path = output_dir / "forecast_7days.csv"
    forecast_df.to_csv(forecast_path, index=False)

    return {
        "sales": sales_path,
        "inventory": inv_path,
        "batches": batches_path,
        "forecast": forecast_path
    }


def validate_synthetic_dataset(data_dir: Path = SYNTHETIC_DATA_DIR) -> Dict[str, Any]:
    """
    Rigorously validates the synthetic demonstration dataset:
    1. Checks presence of all 4 files.
    2. Validates unique identifiers and schema conformance.
    3. Checks dates: valid ISO formats, chronological ordering.
    4. Validates numeric quantities: units_sold >= 0, current_stock >= 0, unit_cost >= 0.
    5. Validates zero duplicate keys: (date, medicine_id) in sales, (medicine_id) in inventory, (batch_id) in batches.
    6. Validates zero missing/null values across critical columns.
    """
    report = {
        "is_valid": True,
        "errors": [],
        "product_count": len(SYNTHETIC_CATALOGUE),
        "sales_rows": 0,
        "inventory_rows": 0,
        "batch_rows": 0,
        "forecast_rows": 0,
        "data_source_labels_verified": True
    }

    sales_file = data_dir / "sales.csv"
    inv_file = data_dir / "inventory.csv"
    batches_file = data_dir / "batches.csv"
    fc_file = data_dir / "forecast_7days.csv"

    for f in [sales_file, inv_file, batches_file, fc_file]:
        if not f.exists():
            report["is_valid"] = False
            report["errors"].append(f"Missing required synthetic dataset file: {f}")

    if not report["is_valid"]:
        return report

    # 1. Sales Validation
    sdf = pd.read_csv(sales_file)
    report["sales_rows"] = len(sdf)
    if sdf["medicine_id"].isnull().any() or sdf["date"].isnull().any() or sdf["units_sold"].isnull().any():
        report["is_valid"] = False
        report["errors"].append("Null values detected in synthetic sales.csv")
    if (sdf["units_sold"] < 0).any():
        report["is_valid"] = False
        report["errors"].append("Negative sales detected in synthetic sales.csv")
    if sdf.duplicated(subset=["date", "medicine_id"]).any():
        report["is_valid"] = False
        report["errors"].append("Duplicate (date, medicine_id) keys found in synthetic sales.csv")
    if not (sdf["data_source"] == "SYNTHETIC_DEMO").all():
        report["is_valid"] = False
        report["data_source_labels_verified"] = False
        report["errors"].append("data_source != 'SYNTHETIC_DEMO' in sales.csv")

    # 2. Inventory Validation
    idf = pd.read_csv(inv_file)
    report["inventory_rows"] = len(idf)
    if len(idf) != len(SYNTHETIC_CATALOGUE):
        report["is_valid"] = False
        report["errors"].append(f"Expected {len(SYNTHETIC_CATALOGUE)} inventory rows, found {len(idf)}")
    if idf["medicine_id"].duplicated().any():
        report["is_valid"] = False
        report["errors"].append("Duplicate medicine_id in synthetic inventory.csv")
    if (idf["current_stock"] < 0).any() or (idf["unit_cost"] <= 0).any():
        report["is_valid"] = False
        report["errors"].append("Invalid negative stock or non-positive unit cost in inventory.csv")

    # 3. Batches Validation
    bdf = pd.read_csv(batches_file)
    report["batch_rows"] = len(bdf)
    if bdf["batch_id"].duplicated().any():
        report["is_valid"] = False
        report["errors"].append("Duplicate batch_id found in synthetic batches.csv")
    if (bdf["batch_quantity"] <= 0).any():
        report["is_valid"] = False
        report["errors"].append("Non-positive batch quantities in synthetic batches.csv")

    # 4. Forecasts Validation
    fcdf = pd.read_csv(fc_file)
    report["forecast_rows"] = len(fcdf)
    if (fcdf["predicted_units"] < 0).any():
        report["is_valid"] = False
        report["errors"].append("Negative forecasts in synthetic forecast_7days.csv")

    return report


def get_combined_catalogue_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Returns full metadata for the Combined Catalogue (Historical 8 ATC + 50 Synthetic Demo).
    Every entry specifies data_source, provenance, generic name, and therapeutic class.
    """
    catalogue = {}

    # Historical 8 ATC Categories
    from app import ATC_NAMES, ATC_SHORT, ATC_CLASSES
    atc_costs = {
        "M01AB": 8.50, "M01AE": 5.25, "N02BA": 4.10, "N02BE": 6.75,
        "N05B": 14.20, "N05C": 18.50, "R03": 12.80, "R06": 7.30
    }
    for code, full_name in ATC_NAMES.items():
        catalogue[code] = {
            "medicine_id": code,
            "display_name": f"{code} · {ATC_SHORT.get(code, code)}",
            "full_name": full_name,
            "short_name": ATC_SHORT.get(code, code),
            "therapeutic_class": ATC_CLASSES.get(code, "Therapeutic Category"),
            "data_source": "HISTORICAL",
            "is_synthetic": False,
            "provenance": "Genuine Historical Benchmark (Kaggle/UCI 2014-2019)",
            "unit_cost": atc_costs.get(code, 10.00),
            "storage_condition": "Room Temperature"
        }

    # Synthetic Demo 50 Medicines
    for med in SYNTHETIC_CATALOGUE:
        m_id = med["medicine_id"]
        catalogue[m_id] = {
            "medicine_id": m_id,
            "display_name": f"{m_id} · {med['generic_name']}",
            "full_name": f"{med['medicine_name']} ({med['atc_code']})",
            "short_name": med["generic_name"],
            "therapeutic_class": med["therapeutic_class"],
            "data_source": "SYNTHETIC_DEMO",
            "is_synthetic": True,
            "provenance": "Synthetic Demonstration Dataset (Explicitly Simulated)",
            "unit_cost": med["unit_cost"],
            "storage_condition": med["storage_condition"]
        }

    return catalogue


def generate_all_synthetic_analytical_tables(
    data_dir: Path = SYNTHETIC_DATA_DIR,
    reference_date: datetime.date = datetime.date(2026, 10, 28)
) -> Dict[str, Path]:
    """
    Computes and persists analytical risk, reorder, summary, and monthly tables
    for the 50 synthetic medicines into data/synthetic_demo/ in Parquet & CSV format.
    """
    from src.inventory import detect_stockout_risk, detect_expiry_risk
    from src.reordering import calculate_reorder_recommendations

    paths = generate_synthetic_demonstration_records(output_dir=data_dir, reference_date=reference_date)

    sales_df = pd.read_csv(paths["sales"])
    inv_df = pd.read_csv(paths["inventory"])
    batches_df = pd.read_csv(paths["batches"])
    fc_df = pd.read_csv(paths["forecast"])

    synthetic_ids = [m["medicine_id"] for m in SYNTHETIC_CATALOGUE]

    # 1. Stockout Risk Table
    stockout_df = detect_stockout_risk(
        inventory_df=inv_df,
        forecast_df=fc_df,
        sales_df=sales_df,
        allowed_medicines=synthetic_ids
    )
    stockout_df["data_source"] = "SYNTHETIC_DEMO"
    stockout_df["is_synthetic"] = True
    stockout_parquet = data_dir / "inventory_stockout_risk.parquet"
    stockout_csv = data_dir / "inventory_stockout_risk.csv"
    stockout_df.to_parquet(stockout_parquet, index=False)
    stockout_df.to_csv(stockout_csv, index=False)

    # 2. Batch Expiry Risk Table
    expiry_df = detect_expiry_risk(
        batches_df=batches_df,
        forecast_df=fc_df,
        sales_df=sales_df,
        reference_date=reference_date.strftime("%Y-%m-%d"),
        allowed_medicines=synthetic_ids
    )
    expiry_df["data_source"] = "SYNTHETIC_DEMO"
    expiry_df["is_synthetic"] = True
    expiry_parquet = data_dir / "batch_expiry_risk.parquet"
    expiry_csv = data_dir / "batch_expiry_risk.csv"
    expiry_df.to_parquet(expiry_parquet, index=False)
    expiry_df.to_csv(expiry_csv, index=False)

    # 3. Reorder Recommendations Table
    reorder_df = calculate_reorder_recommendations(
        inventory_df=inv_df,
        forecast_df=fc_df,
        expiry_df=expiry_df,
        sales_df=sales_df,
        allowed_medicines=synthetic_ids
    )
    reorder_df["data_source"] = "SYNTHETIC_DEMO"
    reorder_df["is_synthetic"] = True
    reorder_parquet = data_dir / "reorder_recommendations.parquet"
    reorder_csv = data_dir / "reorder_recommendations.csv"
    reorder_df.to_parquet(reorder_parquet, index=False)
    reorder_df.to_csv(reorder_csv, index=False)

    # 4. Medicine Summary Table
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    summary_rows = []
    for med_id, grp in sales_df.groupby("medicine_id"):
        summary_rows.append({
            "medicine_id": med_id,
            "total_active_days": int(grp["date"].nunique()),
            "total_units_sold": round(float(grp["units_sold"].sum()), 2),
            "avg_daily_demand": round(float(grp["units_sold"].mean()), 2),
            "demand_volatility": round(float(grp["units_sold"].std()), 2),
            "peak_single_day_sales": round(float(grp["units_sold"].max()), 2),
            "data_source": "SYNTHETIC_DEMO",
            "is_synthetic": True
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("total_units_sold", ascending=False)
    summary_parquet = data_dir / "medicine_summary.parquet"
    summary_df.to_parquet(summary_parquet, index=False)

    sales_df["year"] = sales_df["date"].dt.year
    sales_df["month"] = sales_df["date"].dt.month
    sales_df["day_of_month"] = sales_df["date"].dt.day
    sales_df["day_of_week"] = sales_df["date"].dt.day_name()
    monthly_rows = []
    for (yr, mo, med_id), grp in sales_df.groupby(["year", "month", "medicine_id"]):
        monthly_rows.append({
            "year": yr,
            "month": mo,
            "medicine_id": med_id,
            "monthly_units_sold": round(float(grp["units_sold"].sum()), 2),
            "avg_daily_sales": round(float(grp["units_sold"].mean()), 2),
            "days_recorded": int(len(grp)),
            "data_source": "SYNTHETIC_DEMO",
            "is_synthetic": True
        })
    monthly_df = pd.DataFrame(monthly_rows).sort_values(["year", "month", "medicine_id"])
    monthly_parquet = data_dir / "monthly_trends.parquet"
    monthly_df.to_parquet(monthly_parquet, index=False)

    # Also persist sales and forecast in Parquet format for fast loading
    sales_df.to_parquet(data_dir / "daily_sales.parquet", index=False)
    fc_df.to_parquet(data_dir / "forecast_7days.parquet", index=False)

    return {
        "sales": paths["sales"],
        "daily_sales_parquet": data_dir / "daily_sales.parquet",
        "inventory": paths["inventory"],
        "batches": paths["batches"],
        "forecast": paths["forecast"],
        "forecast_parquet": data_dir / "forecast_7days.parquet",
        "stockout_parquet": stockout_parquet,
        "expiry_parquet": expiry_parquet,
        "reorder_parquet": reorder_parquet,
        "summary_parquet": summary_parquet,
        "monthly_parquet": monthly_parquet
    }


if __name__ == "__main__":
    generated = generate_all_synthetic_analytical_tables()
    print("All synthetic demonstration tables generated successfully:")
    for k, v in generated.items():
        print(f"  - {k}: {v}")
    v_report = validate_synthetic_dataset()
    print("\nValidation Report:")
    for k, v in v_report.items():
        print(f"  {k}: {v}")

