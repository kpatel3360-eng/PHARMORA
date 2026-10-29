"""
Unit and Integration Test Suite: PHARMORA Product Catalogue Expansion
Validates:
1. Preservation of original baseline data/salesdaily.csv (checksum & row count).
2. Physical separation of synthetic demo dataset in data/synthetic_demo/.
3. Exact 50 synthetic medicine products across 8 clinical therapeutic classes.
4. Schema, date formats, non-negativity, zero duplicates, zero nulls.
5. Zero-fabrication guarantee: ML holdout evaluations are strictly reserved for genuine historical data.
6. Analytical engines (Stockout risk, Expiry audit, Smart reordering) functionality on synthetic catalogue.
7. Data source loader reactivity: Historical (8/9), Synthetic Demo (50), Combined (58/59).
"""

import unittest
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

from src.synthetic_demo import (
    SYNTHETIC_MEDICINES,
    SYNTHETIC_DIR,
    validate_synthetic_dataset,
    generate_synthetic_demonstration_records,
    generate_all_synthetic_analytical_tables
)
from src.inventory import detect_stockout_risk, detect_expiry_risk
from src.reordering import calculate_reorder_recommendations


class TestCatalogueExpansion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure synthetic analytical tables exist
        generate_all_synthetic_analytical_tables()

    def test_01_preservation_of_original_salesdaily(self):
        """Requirement 1: Verify data/salesdaily.csv is completely preserved."""
        salesdaily_path = Path("data/salesdaily.csv")
        self.assertTrue(salesdaily_path.exists(), "data/salesdaily.csv must exist")

        df = pd.read_csv(salesdaily_path)
        self.assertEqual(len(df), 2106, "data/salesdaily.csv must have exactly 2,106 rows")
        expected_cols = ['datum', 'M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06', 'Year', 'Month', 'Hour', 'Weekday Name']
        self.assertListEqual(list(df.columns), expected_cols, "data/salesdaily.csv columns must not be modified")

        # Verify total observational volume (~127,595.50 units)
        total_vol = df[["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]].sum().sum()
        self.assertAlmostEqual(total_vol, 127595.5037, places=3,
                               msg="Total volume in data/salesdaily.csv must match original benchmark")

    def test_02_synthetic_demo_separate_storage(self):
        """Requirement 4: Verify demonstration data is stored separately from historical data."""
        syn_dir = Path("data/synthetic_demo")
        self.assertTrue(syn_dir.exists(), "data/synthetic_demo/ directory must exist separately")

        # Verify key separate artifacts exist
        expected_files = [
            syn_dir / "sales.csv",
            syn_dir / "daily_sales.parquet",
            syn_dir / "inventory.csv",
            syn_dir / "batches.csv",
            syn_dir / "forecast_7days.parquet",
            syn_dir / "inventory_stockout_risk.parquet",
            syn_dir / "batch_expiry_risk.parquet",
            syn_dir / "reorder_recommendations.parquet"
        ]
        for f in expected_files:
            self.assertTrue(f.exists(), f"Synthetic artifact {f.name} must exist in {syn_dir}")

    def test_03_synthetic_catalogue_dimensions_and_classes(self):
        """Requirement 3: Verify approximately 50 medicine products spanning therapeutic classes."""
        catalogue = SYNTHETIC_MEDICINES
        self.assertEqual(len(catalogue), 50, "Synthetic catalogue must contain exactly 50 products")

        # Verify unique IDs
        ids = [m["medicine_id"] for m in catalogue]
        self.assertEqual(len(set(ids)), 50, "All 50 synthetic medicine IDs must be unique")
        for mid in ids:
            self.assertTrue(mid.startswith("SYN-"), f"Synthetic ID {mid} must have SYN- prefix")

        # Verify therapeutic diversity (at least 8 clinical classes)
        classes = set(m["therapeutic_class"] for m in catalogue)
        self.assertGreaterEqual(len(classes), 8, "Must span at least 8 therapeutic classes")
        self.assertIn("Broad Spectrum Penicillins", classes)
        self.assertIn("HMG-CoA Reductase Inhibitors", classes)
        self.assertIn("Biguanides / Oral Hypoglycemics", classes)
        self.assertIn("Proton Pump Inhibitors", classes)

    def test_04_synthetic_sales_data_integrity(self):
        """Requirement 9: Validate identifiers, dates, numeric quantities, duplicate records and missing values."""
        val_result = validate_synthetic_dataset()
        self.assertTrue(val_result["is_valid"], f"Synthetic dataset validation failed: {val_result.get('errors')}")
        self.assertEqual(val_result["errors"], [], "Validation errors must be empty")
        self.assertEqual(val_result["product_count"], 50, "Must validate 50 unique medicines")
        self.assertEqual(val_result["sales_rows"], 9000, "50 meds * 180 days must equal 9,000 sales rows")

        # Load sales.csv and inspect manually
        sales_df = pd.read_csv(SYNTHETIC_DIR / "sales.csv")
        self.assertEqual(sales_df["units_sold"].isna().sum(), 0, "No null values allowed in sales")
        self.assertTrue((sales_df["units_sold"] >= 0).all(), "Units sold must be non-negative")
        self.assertTrue((sales_df["data_source"] == "SYNTHETIC_DEMO").all(), "All rows must be labelled SYNTHETIC_DEMO")

        # Check duplicate (date, medicine_id)
        dup_count = sales_df.duplicated(subset=["date", "medicine_id"]).sum()
        self.assertEqual(dup_count, 0, "Zero duplicates allowed in synthetic sales")

    def test_05_zero_fabrication_forecasting_evaluations(self):
        """Requirement 7: Run forecasting evaluation only on valid historical observations."""
        # Genuine evaluations table
        eval_path = Path("processed/evaluations.parquet")
        if eval_path.exists():
            eval_df = pd.read_parquet(eval_path)
            # Ensure no SYN- products exist in genuine evaluations table
            syn_in_eval = eval_df[eval_df["medicine_id"].str.startswith("SYN-")]
            self.assertEqual(len(syn_in_eval), 0,
                             "Zero synthetic products may have fabricated holdout evaluation metrics")

        # Synthetic forecast table
        syn_fc_path = SYNTHETIC_DIR / "forecast_7days.parquet"
        self.assertTrue(syn_fc_path.exists())
        syn_fc = pd.read_parquet(syn_fc_path)
        self.assertEqual(len(syn_fc), 350, "50 products * 7 forecast days = 350 rows")
        self.assertTrue(syn_fc["model_name"].str.contains("Simulated", case=False).all(),
                        "Synthetic forecasts must be explicitly labelled with simulated method")
        self.assertTrue((syn_fc["is_synthetic"] == True).all(),
                        "is_synthetic flag must be True for all demo forecasts")

    def test_06_analytical_engines_support_synthetic_products(self):
        """Requirement 8: Ensure stockout, reorder and inventory engines process synthetic products safely."""
        syn_inv = pd.read_csv(SYNTHETIC_DIR / "inventory.csv")
        syn_batches = pd.read_csv(SYNTHETIC_DIR / "batches.csv")
        syn_sales = pd.read_parquet(SYNTHETIC_DIR / "daily_sales.parquet")
        syn_fc = pd.read_parquet(SYNTHETIC_DIR / "forecast_7days.parquet")
        allowed_syn_meds = [m["medicine_id"] for m in SYNTHETIC_MEDICINES]

        # Stockout risk engine
        so_df = detect_stockout_risk(
            inventory_df=syn_inv,
            forecast_df=syn_fc,
            sales_df=syn_sales,
            allowed_medicines=allowed_syn_meds
        )
        self.assertEqual(len(so_df), 50, "Stockout risk must compute for all 50 synthetic medicines")
        self.assertIn("risk_status", so_df.columns)
        self.assertIn("reorder_point", so_df.columns)

        # Batch expiry audit engine
        exp_df = detect_expiry_risk(
            batches_df=syn_batches,
            forecast_df=syn_fc,
            sales_df=syn_sales,
            reference_date="2026-10-28",
            allowed_medicines=allowed_syn_meds
        )
        self.assertEqual(len(exp_df), 120, "Batch audit must compute for all 120 synthetic batches")
        self.assertIn("expiry_status", exp_df.columns)
        self.assertIn("value_at_risk", exp_df.columns)

        # Reorder recommendations engine
        reorder_df = calculate_reorder_recommendations(
            inventory_df=syn_inv,
            forecast_df=syn_fc,
            expiry_df=exp_df,
            sales_df=syn_sales,
            allowed_medicines=allowed_syn_meds
        )
        self.assertEqual(len(reorder_df), 50, "Reorder recommendations must compute for all 50 synthetic medicines")
        self.assertIn("suggested_order_quantity", reorder_df.columns)
        self.assertIn("reorder_status", reorder_df.columns)

    def test_07_app_data_loaders_and_catalogue_counts(self):
        """Requirement 5 & 6: Data source loaders return correct catalogue counts."""
        from app import (
            load_all_pharmora_data,
            load_synthetic_demo_data,
            load_combined_pharmora_data
        )

        # 1. Historical dataset
        hist_data = load_all_pharmora_data()
        hist_meds = hist_data["daily"]["medicine_id"].unique()
        self.assertGreaterEqual(len(hist_meds), 8, "Historical must contain at least 8 baseline ATC categories")
        for mid in hist_meds:
            self.assertFalse(mid.startswith("SYN-"), f"Historical dataset must not contain synthetic ID {mid}")

        # 2. Synthetic demo dataset
        syn_data = load_synthetic_demo_data()
        syn_meds = syn_data["daily"]["medicine_id"].unique()
        self.assertEqual(len(syn_meds), 50, "Synthetic dataset must contain exactly 50 products")
        for mid in syn_meds:
            self.assertTrue(mid.startswith("SYN-"), f"Synthetic dataset ID {mid} must have SYN- prefix")
        self.assertEqual(len(syn_data["daily"]), 9000, "Synthetic daily sales must be exactly 9,000 rows")
        self.assertEqual(len(syn_data["stockout"]), 50, "Synthetic stockout must be exactly 50 rows")
        self.assertEqual(len(syn_data["expiry"]), 120, "Synthetic expiry must be exactly 120 batch rows")
        self.assertEqual(len(syn_data["reorder"]), 50, "Synthetic reorder must be exactly 50 rows")

        # 3. Combined product catalogue
        comb_data = load_combined_pharmora_data()
        comb_meds = comb_data["daily"]["medicine_id"].unique()
        expected_combined_count = len(hist_meds) + 50
        self.assertEqual(len(comb_meds), expected_combined_count,
                         f"Combined catalogue must contain {expected_combined_count} products (Historical + 50 Synthetic)")

        # Verify total combined sales rows
        expected_rows = len(hist_data["daily"]) + 9000
        self.assertEqual(len(comb_data["daily"]), expected_rows,
                         f"Combined sales must equal historical ({len(hist_data['daily'])}) + synthetic (9,000)")

        # Verify data_source column is present in combined
        self.assertIn("data_source", comb_data["daily"].columns)
        self.assertIn("data_source", comb_data["stockout"].columns)
        self.assertIn("data_source", comb_data["expiry"].columns)
        self.assertIn("data_source", comb_data["reorder"].columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
