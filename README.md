# 💊 PHARMORA: AI-Powered Pharmacy Intelligence Platform

A Big Data Analytics engineering project designed for pharmaceutical demand forecasting, large-scale transaction processing, and automated inventory optimization.

**Project Milestone**: Phase 5 Complete & Verified (End-to-End PySpark Pipeline, Demand Forecasting, Inventory & FEFO Expiry Risk, Prescriptive Smart Reordering, Streamlit Intelligence Dashboard & System Verification Suite)  
**Tech Stack**: Python 3.10+, Apache Spark (PySpark 3.5 & Spark SQL), Apache Parquet, Scikit-learn, Streamlit, Pandas, Plotly.

---

## 📁 Project Architecture & Structure

```
Pharmora/
│
├── README.md                # Comprehensive project documentation & runbook
├── requirements.txt         # Project dependencies specification
├── verify_pipeline.py       # Data pipeline validation & reconciliation script
├── app.py                   # Streamlit intelligence dashboard
│
├── data/                    # Transactional, warehouse inventory, and batch datasets
│   ├── salesdaily.csv       # Real daily pharmaceutical sales (2014-2019, 2,106 rows)
│   ├── imported_sales.csv   # Staging location for supplementary imported sales records
│   ├── inventory.csv        # Synthetic warehouse inventory records (8 medicines)
│   ├── batches.csv          # Synthetic pharmaceutical batch tracking records (16 batches)
│   └── .gitkeep
│
├── hadoop/                  # Windows native Hadoop binaries
│   └── bin/
│       ├── hadoop.dll       # Windows native library for FileOutputCommitter
│       └── winutils.exe     # Hadoop binary utility
│
├── processed/               # Production Parquet tables output by PySpark & ML
│   ├── daily_sales.parquet             # Normalized long-format facts (16,848+ rows)
│   ├── medicine_summary.parquet        # Spark SQL aggregated KPIs per drug (8+ rows)
│   ├── monthly_trends.parquet          # Year-month aggregate sales trajectories (560+ rows)
│   ├── forecast_7days.parquet          # 7-day recursive daily demand forecast (56 rows)
│   ├── forecast_7days.csv              # CSV export of 7-day future predictions
│   ├── forecast_evaluations.parquet    # Chronological holdout evaluation metrics (8 rows)
│   ├── forecast_evaluations.csv        # CSV export of model performance metrics
│   ├── inventory_stockout_risk.parquet # Stockout risk & ROP indicators (8 rows)
│   ├── inventory_stockout_risk.csv     # CSV export of stockout risk analysis
│   ├── batch_expiry_risk.parquet       # FEFO batch expiration risk audit (16 rows)
│   ├── batch_expiry_risk.csv           # CSV export of batch expiry audit
│   ├── reorder_recommendations.parquet # Prescriptive replenishment recommendations (8 rows)
│   └── reorder_recommendations.csv     # CSV export of reorder recommendations
│
├── models/                  # Serialized machine learning models (Joblib)
│   ├── rf_M01AB.joblib      # Random Forest forecaster for Diclofenac
│   ├── rf_M01AE.joblib      # Random Forest forecaster for Ibuprofen
│   ├── rf_N02BA.joblib      # Random Forest forecaster for Aspirin
│   ├── rf_N02BE.joblib      # Random Forest forecaster for Paracetamol
│   ├── rf_N05B.joblib       # Random Forest forecaster for Anxiolytics
│   ├── rf_N05C.joblib       # Random Forest forecaster for Hypnotics/Sedatives
│   ├── rf_R03.joblib        # Random Forest forecaster for Asthma medication
│   └── rf_R06.joblib        # Random Forest forecaster for Antihistamines
│
├── test_import_sales_feature.py # Test suite for CSV validation, deduplication & Parquet updates
├── test_selector_reactivity.py  # Test suite for medicine selector reactivity & KPIs
├── verify_project_end_to_end.py # Comprehensive end-to-end integration verification suite
│
└── src/                     # Core analytical modules
    ├── __init__.py
    ├── data_pipeline.py     # PySpark pipeline: Ingestion, schema validation, unpivot, Spark SQL
    ├── data_ingestion.py    # Sales Ingestion Engine: Schema validation, deduplication, incremental updates
    ├── forecasting.py       # Time-series ML: Lags, leakage-safe features, RF & Baseline, 7-day recursive
    ├── inventory.py         # Inventory Risk Engine: Stockout detection, FEFO expiry risk audit
    └── reordering.py        # Smart Reordering Engine: Prescriptive order quantities, timing, and clinical flags
```

---

## 📊 Dataset: ATC Classifications & Structure

The raw transactional dataset (`data/salesdaily.csv`) contains daily sales aggregated by **Anatomical Therapeutic Chemical (ATC)** classification:

- **File Location**: `data/salesdaily.csv`
- **Total Records**: 2,106 rows
- **Date Range**: `2014-01-02` to `2019-10-08` (2,106 consecutive days)
- **Raw Columns (13)**: `datum`, `M01AB`, `M01AE`, `N02BA`, `N02BE`, `N05B`, `N05C`, `R03`, `R06`, `Year`, `Month`, `Hour`, `Weekday Name`
- **Date Format**: `M/d/yyyy` (e.g., `1/2/2014`)

### Medicine Categories

| ATC Code | Drug Classification | Real-World Examples | Total Units Sold | Avg Daily Demand | Demand Volatility (StdDev) |
|:---|:---|:---|:---:|:---:|:---:|
| **N02BE** | Analgesics & Antipyretics (Pyrazolones & Anilides) | Paracetamol (Acetaminophen) | 63,005.40 | 29.92 | 15.59 |
| **N05B**  | Psycholeptics (Anxiolytic agents) | Diazepam, Alprazolam | 18,645.74 | 8.85 | 5.61 |
| **R03**   | Obstructive Airway Disease Agents | Salbutamol (Albuterol) | 11,608.82 | 5.51 | 6.43 |
| **M01AB** | Anti-inflammatory (Acetic acid derivatives) | Diclofenac, Indomethacin | 10,600.94 | 5.03 | 2.74 |
| **M01AE** | Anti-inflammatory (Propionic acid derivatives) | Ibuprofen, Naproxen | 8,204.62 | 3.90 | 2.13 |
| **N02BA** | Analgesics & Antipyretics (Salicylic acid) | Aspirin | 8,172.21 | 3.88 | 2.38 |
| **R06**   | Antihistamines for Systemic Use | Cetirizine, Loratadine | 6,107.82 | 2.90 | 2.42 |
| **N05C**  | Psycholeptics (Hypnotics & Sedatives) | Zolpidem, Temazepam | 1,249.96 | 0.59 | 1.09 |
| **TOTAL** | *All 8 Therapeutic Classes Combined* | — | **127,595.50** | **60.59** | — |

---

## ⚙️ Environment Prerequisites & Dependencies

### Prerequisites
1. **Python**: Version 3.10+ (tested on Python 3.14 / 3.13)
2. **Java Runtime**: OpenJDK 17 LTS (required for PySpark / Spark SQL execution)
3. **Windows Hadoop Binaries**: `winutils.exe` and `hadoop.dll` are pre-configured in `hadoop/bin/` and automatically loaded by the pipeline.

### Installation
Install required packages into your active Python environment:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the PySpark Data Pipeline (Phase 1)

```bash
python -m src.data_pipeline
```
*Or directly by file path:*
```bash
python src/data_pipeline.py
```

### Automated Pipeline Verification
```bash
python verify_pipeline.py
```
- **Input / Output Row Match**: 2,106 raw rows $\to$ 16,848 processed long facts ($2,106 \times 8$).
- **Reconciliation Discrepancy**: `0.000000` (127,595.5037 raw units $\to$ 127,595.5037 processed units).
- **Quality Checks**: 0 null dates, 0 negative values, 0 duplicate keys.

---

## 🔮 Phase 2: Medicine Demand Forecasting Engine

```bash
python -m src.forecasting
```
*Or directly by file path:*
```bash
python src/forecasting.py
```

### Chronological Test Set Comparison (60 Holdout Days)

| Medicine | ATC Classification | Mean Daily Demand | Baseline MAE | **Random Forest MAE** | Baseline RMSE | **Random Forest RMSE** | Baseline WAPE (%) | **Random Forest WAPE (%)** | MAE Improvement | Winning Model |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **M01AB** | Anti-inflammatory (Diclofenac) | 5.52 | 2.488 | **2.350** | 2.999 | **2.839** | 45.08% | **42.58%** | +5.56% | **Random Forest** |
| **M01AE** | Anti-inflammatory (Ibuprofen) | 3.59 | 1.636 | **1.543** | 2.216 | **2.022** | 45.59% | **42.99%** | +5.71% | **Random Forest** |
| **N02BA** | Analgesics (Aspirin) | 2.91 | 1.292 | **1.156** | 1.615 | **1.530** | 44.37% | **39.69%** | +10.55% | **Random Forest** |
| **N02BE** | Analgesics/Antipyretics (Paracetamol) | 27.52 | 7.990 | **6.995** | 10.878 | **9.916** | 29.03% | **25.41%** | +12.46% | **Random Forest** |
| **N05B**  | Psycholeptics (Anxiolytics) | 8.01 | 3.370 | **3.036** | 4.108 | **3.824** | 42.05% | **37.89%** | +9.91% | **Random Forest** |
| **N05C**  | Hypnotics & Sedatives (Sleep aids) | 0.60 | 0.876 | **0.804** | 1.087 | **1.011** | 146.03% | **133.95%** | +8.27% | **Random Forest** |
| **R03**   | Asthma / Airway Disease | 3.82 | 3.953 | **3.810** | 5.598 | **5.322** | 103.39% | **99.65%** | +3.61% | **Random Forest** |
| **R06**   | Antihistamines (Allergies) | 3.16 | 1.735 | **1.490** | 2.226 | **2.041** | 54.82% | **47.08%** | +14.12% | **Random Forest** |
| **AVERAGE**| *All 8 Drug Categories* | **6.89** | **2.917** | **2.648** | **3.841** | **3.563** | — | — | **+9.24%** | **Random Forest (8/8)** |

---

## 📦 Phase 3: Inventory Risk Engine (Stockout & Expiry Intelligence)

```bash
python -m src.inventory
```
*Or directly by file path:*
```bash
python src/inventory.py
```

### Stockout & Expiry Summary
- **Stockout Audit**: 2 Critical Stockout Risks (`M01AB`, `N02BE`), 2 Reorder Warnings (`M01AE`, `R03`), 3 Healthy (`N02BA`, `N05B`, `R06`), 1 Overstocked (`N05C`).
- **Batch Expiry Audit (FEFO Allocation)**: 5 Expired batches (₹898.75) quarantined, 4 Near-Expiry risk batches (₹544.42) flagged for expedited rotation, 7 Healthy batches. Total capital at risk: **₹1,443.17**.

---

## 🛒 Phase 4: Smart Reordering Engine (Prescriptive Procurement)

The Smart Reordering Engine in [`src/reordering.py`](file:///c:/Pharmora/src/reordering.py) synthesizes machine-learning forecasts, warehouse stock levels, pending supplier orders, and batch expiry audits to recommend:
1. **What medicine to reorder** (`medicine_id`)
2. **How much to order** (`suggested_order_quantity`)
3. **When to place the order** (`recommended_order_timing` and `reorder_status`)
4. **Actionable clinical & operational rationale** (`recommendation_reason`)

### Execution Command
Run the reordering module standalone:
```bash
python -m src.reordering
```
*Or directly by file path:*
```bash
python src/reordering.py
```

Optional CLI parameters:
```bash
python src/reordering.py --review_period 7 --inventory data/inventory.csv --forecast processed/forecast_7days.parquet --expiry processed/batch_expiry_risk.parquet
```

---

### Prescriptive Mathematical Framework

1. **Inventory Position (Net Usable Stock)**:
   $$\text{Inventory Position} = \text{Current Stock} + \text{Pending Order Quantity}$$
   *Prevents double-ordering when replenishment shipments are already in transit.*

2. **Forecasted Lead-Time Demand**:
   Demand expected during supplier replenishment lead time ($L$ days):
   $$\text{Lead-Time Demand} = \sum_{t=1}^{\min(L, 7)} \hat{y}_t + \max(0, L - 7) \times \bar{y}_{\text{forecast}}$$

3. **Reorder Point (ROP)**:
   The stock threshold below which an order must be placed:
   $$\text{Reorder Point} = \text{Lead-Time Demand} + \text{Safety Stock}$$

4. **Target Stock (Order-Up-To Level, $S$)**:
   The inventory needed to cover both supplier lead time ($L$) and the periodic review window ($R = 7\text{ days}$):
   $$\text{Target Stock} = \text{Forecast Demand over } (L + R) + \text{Safety Stock}$$

5. **Suggested Order Quantity**:
   $$\text{Suggested Order Quantity} = \begin{cases} \lceil \max(0, \text{Target Stock} - \text{Inventory Position}) \rceil, & \text{if } \text{Inventory Position} \le \text{Reorder Point} \\ 0, & \text{if } \text{Inventory Position} > \text{Reorder Point} \end{cases}$$
   *Medicines with sufficient inventory receive 0 units to prevent unnecessary capital tie-up.*

6. **Order Timing & Urgency Horizon**:
   - **`URGENT`**: $\text{Inventory Position} \le \text{Reorder Point}$.
     - `IMMEDIATE (Within 24 Hours - Stockout Imminent)`: If $\text{Inventory Position} < \text{Lead-Time Demand}$.
     - `IMMEDIATE (Current Order Cycle)`: If $\text{Lead-Time Demand} \le \text{Inventory Position} \le \text{Reorder Point}$.
   - **`UPCOMING`**: $\text{Inventory Position} > \text{Reorder Point}$ AND $\text{Days to ROP} \le R$.
     $$\text{Days until ROP} = \frac{\text{Inventory Position} - \text{Reorder Point}}{\text{Daily Demand Rate}}$$
     Timing: `UPCOMING (In ~X Days)`.
   - **`NOT_REQUIRED`**: $\text{Days until ROP} > R$ or overstocked. Timing: `NOT_REQUIRED`.

7. **Healthcare Compliance & Expiry Risk Safeguard**:
   - If expired batches are detected in the warehouse, the engine explicitly alerts:
     `EXPIRED STOCK DETECTED: X units expired. Must be quarantined immediately for disposal; NEVER sell expired medicines.`
   - If near-expiry batches are detected, the engine flags:
     `EXPIRY RISK REVIEW: Y units at risk of expiring unsold before consumption. Verify batch FEFO status before placing large replenishment orders.`

---

### Actual Smart Reordering Recommendations Output

| Medicine | ATC Classification | Current Stock | Pending Order | Lead-Time Demand | Reorder Point (ROP) | Target Stock | Suggested Order Qty | Reorder Status | Recommended Timing | Clinical & Operational Recommendation Reason |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **M01AB** | Anti-inflammatory (Diclofenac) | 18.0 | 0.0 | 34.95 | 46.95 | 81.90 | **64 units** | `URGENT` | **IMMEDIATE (Within 24 Hours - Stockout Imminent)** | **CRITICAL**: Inventory position (18.0 units) cannot cover lead-time demand (35.0 units). Stockout guaranteed before replenishment arrives. **EXPIRED STOCK DETECTED**: 8.0 expired units found in warehouse. Must be quarantined immediately for disposal; NEVER sell expired medicines. |
| **M01AE** | Anti-inflammatory (Ibuprofen) | 26.0 | 10.0 | 29.82 | 38.82 | 68.64 | **33 units** | `URGENT` | **IMMEDIATE (Current Order Cycle)** | **REORDER THRESHOLD BREACHED**: Inventory position (36.0 units) is at or below ROP (38.8 units). Safety stock buffer (9.0 units) is being drawn down. In-flight pending order of 10 units accounted for to prevent duplicate ordering. **EXPIRY RISK REVIEW**: 7.1 units across active batches are at risk of expiring unsold before consumption. Verify batch FEFO status before placing large replenishment orders. |
| **N02BA** | Analgesics (Aspirin) | 45.0 | 0.0 | 14.76 | 24.76 | 46.33 | **0 units** | `UPCOMING` | **UPCOMING (In ~6.7 Days)** | Sufficient for immediate needs, but inventory position (45.0 units) will deplete to ROP (24.8 units) in approximately 6.7 days (within 7-day review window). |
| **N02BE** | Analgesics (Paracetamol) | 135.0 | 50.0 | 252.13 | 320.13 | 572.26 | **388 units** | `URGENT` | **IMMEDIATE (Within 24 Hours - Stockout Imminent)** | **CRITICAL**: Inventory position (185.0 units) cannot cover lead-time demand (252.1 units). Stockout guaranteed before replenishment arrives. In-flight pending order of 50 units accounted for to prevent duplicate ordering. **EXPIRED STOCK DETECTED**: 25.0 expired units found in warehouse. Must be quarantined immediately for disposal; NEVER sell expired medicines. |
| **N05B**  | Psycholeptics (Anxiolytics) | 120.0 | 0.0 | 64.42 | 88.42 | 152.84 | **0 units** | `UPCOMING` | **UPCOMING (In ~3.4 Days)** | Sufficient for immediate needs, but inventory position (120.0 units) will deplete to ROP (88.4 units) in approximately 3.4 days (within 7-day review window). **EXPIRED STOCK DETECTED**: 15.0 expired units found in warehouse. Must be quarantined immediately for disposal; NEVER sell expired medicines. **EXPIRY RISK REVIEW**: 7.9 units across active batches are at risk of expiring unsold before consumption. Verify batch FEFO status before placing large replenishment orders. |
| **N05C**  | Hypnotics (Sleep aids) | 38.0 | 0.0 | 4.30 | 9.30 | 13.60 | **0 units** | `NOT_REQUIRED` | **NOT_REQUIRED** | Optimal inventory buffer (38.0 units). Covers demand for ~46.7 days beyond ROP. **EXPIRED STOCK DETECTED**: 10.0 expired units found in warehouse. Must be quarantined immediately for disposal; NEVER sell expired medicines. **EXPIRY RISK REVIEW**: 13.3 units across active batches are at risk of expiring unsold before consumption. Verify batch FEFO status before placing large replenishment orders. |
| **R03**   | Asthma Medication | 45.0 | 25.0 | 62.81 | 90.81 | 134.78 | **65 units** | `URGENT` | **IMMEDIATE (Current Order Cycle)** | **REORDER THRESHOLD BREACHED**: Inventory position (70.0 units) is at or below ROP (90.8 units). Safety stock buffer (28.0 units) is being drawn down. In-flight pending order of 25 units accounted for to prevent duplicate ordering. **EXPIRED STOCK DETECTED**: 12.0 expired units found in warehouse. Must be quarantined immediately for disposal; NEVER sell expired medicines. **EXPIRY RISK REVIEW**: 6.8 units across active batches are at risk of expiring unsold before consumption. Verify batch FEFO status before placing large replenishment orders. |
| **R06**   | Antihistamines (Allergies) | 35.0 | 0.0 | 14.88 | 25.88 | 40.76 | **0 units** | `UPCOMING` | **UPCOMING (In ~4.3 Days)** | Sufficient for immediate needs, but inventory position (35.0 units) will deplete to ROP (25.9 units) in approximately 4.3 days (within 7-day review window). |

#### Reorder Summary:
- **Procurement Urgency**: **4 URGENT** (`M01AB`, `M01AE`, `N02BE`, `R03`) | **3 UPCOMING** (`N02BA`, `N05B`, `R06`) | **1 NOT REQUIRED** (`N05C`).
- **Total Recommended Replenishment Volume**: **550 units** across the 4 urgent medicines.
- **Pending Order Reconciliation**:
  - `N02BE`: Needed 438 units to reach target stock; having 50 units already pending reduced the order to **388 units**.
  - `M01AE`: Needed 43 units; having 10 units pending reduced the order to **33 units**.
  - `R03`: Needed 90 units; having 25 units pending reduced the order to **65 units**.

---

## 🗄️ Output Schemas & Storage Locations

### Reorder Recommendations Table: `processed/reorder_recommendations.parquet` (8 rows)
```
root
 |-- medicine_id: string (nullable = true)
 |-- current_stock: double (nullable = true)
 |-- pending_order_quantity: double (nullable = true)
 |-- forecast_lead_time_demand: double (nullable = true)
 |-- safety_stock: double (nullable = true)
 |-- reorder_point: double (nullable = true)
 |-- target_stock: double (nullable = true)
 |-- suggested_order_quantity: integer (nullable = true)
 |-- reorder_status: string (nullable = true)
 |-- recommended_order_timing: string (nullable = true)
 |-- recommendation_reason: string (nullable = true)
```
*Also exported as [`processed/reorder_recommendations.csv`](file:///c:/Pharmora/processed/reorder_recommendations.csv).*

---

## ⚠️ Known Reordering Engine Assumptions & Limitations

1. **Discrete Packaging / MOQ**: Suggested order quantities are ceiled to discrete units. In pharmaceutical distribution where suppliers mandate minimum case packs (e.g., 20 or 50 packs), MOQs should be configured in the supplier master.
2. **Fixed Lead Times**: Lead times are treated as deterministic; stochastic supplier delays can be mitigated by expanding statistical safety stock ($z \times \sqrt{L \sigma_d^2 + d^2 \sigma_L^2}$).
3. **Single Supplier per Drug**: The engine assumes a single primary distributor per ATC category. Multiple vendor routing can be layered on top of the target stock calculation.

---

## 🖥️ Phase 5: Interactive Web Dashboard & Demonstration Suite

The executive and operational front-end is implemented in [`app.py`](file:///c:/Pharmora/app.py) using **Streamlit** and **Plotly**. It connects directly to the curated Parquet and CSV artifacts generated by PySpark, Scikit-learn, and the inventory engines.

### Launching the Dashboard
```bash
streamlit run app.py
```
*Or using the Python module runner:*
```bash
python -m streamlit run app.py
```
Access the dashboard in your web browser at: `http://localhost:8501`.

---

### Dashboard Functional Sections

The application is structured into **6 specialized analytical sections**:

1. **📊 Overview (Executive Operations Cockpit)**:
   - High-level KPIs: Total Historical Volume (127,596 units), Warehouse Stock (462 units), Total Inventory Asset Value (₹5,055.25), Capital at Expiry Risk (₹1,443.20), and Urgent Replenishment Demand (550 units).
   - Urgent procurement alerts and healthcare quarantine notifications.
   - Horizontal bar chart of cumulative sales across all 8 ATC categories.
   - Donut chart depicting the distribution of warehouse risk classifications (2 Critical, 2 Warning, 3 Healthy, 1 Overstocked).
   - Holistic drug portfolio diagnostic table combining stock positions and recommended order quantities.

2. **🔮 Medicine Demand Forecast (Machine Learning)**:
   - Dynamic ATC category filter with common drug names (e.g., Paracetamol, Ibuprofen, Salbutamol).
   - Interactive historical actuals vs. 7-day multi-step recursive Random Forest forecast line chart.
   - Evaluation metrics card: Actual mean daily demand, RF MAE, Baseline MAE, RMSE, and WAPE error percentage.
   - Comprehensive 7-day forecast matrix across all 8 medicine categories.
   - Interactive on-the-fly backtesting tool: Allows users to adjust the chronological holdout window (30-120 days) and re-evaluate model performance live.

3. **⚠️ Stockout Risk (Inventory Risk Engine)**:
   - KPI metrics for Critical Stockouts, Reorder Threshold Warnings, Healthy Buffers, and Overstocked medicines.
   - Grouped bar chart comparing **Net Inventory Position vs. Lead-Time Demand vs. Reorder Point (ROP)**.
   - Diagnostic register detailing days of stock cover, safety stock buffers, and actionable recommendations.
   - Interactive What-If Inventory Health Simulator: Test how supplier delays (1-30 days) and simulated stock changes impact safety stock and ROP in real time.

4. **⏳ Expiry Risk (FEFO Batch Audit)**:
   - Synthetic demonstration disclaimer clearly labelling the 16 batches.
   - Critical Healthcare Safety Protocol: Strict prohibition on selling expired pharmaceuticals.
   - Audit metrics: 5 Expired batches (₹898.75 loss), 4 Near-Expiry risk batches (₹544.45 exposed), and 7 Safe batches.
   - Bar chart of financial capital at risk aggregated by medicine category.
   - Scatter plot of batch shelf-life horizon (days until expiration vs. batch size) with today's reference horizon marked.
   - Detailed batch audit table showing days remaining, FEFO allocated demand, unsold units, and quarantine directives.

5. **🛒 Smart Reordering (Prescriptive Procurement)**:
   - Procurement urgency breakdown: 4 URGENT, 3 UPCOMING, 1 NOT REQUIRED.
   - Stacked bar chart showing Current Stock + In-Flight Pending Orders + Suggested Order Quantity against Target Stock ($S$).
   - Reorder Decision Matrix showing exact quantities and timings.
   - Clinical & operational rationale log explaining the exact mathematical derivation for every decision, including pending order deductions and expired batch quarantine notices.

6. **⚡ Big Data Analytics (PySpark & Spark SQL)**:
   - Summary of distributed PySpark pipeline execution (2,106 daily rows $\to$ 16,848 normalized fact rows with 0.000000 unit discrepancy).
   - Day-of-week retail purchasing pattern analysis showing weekend demand surges.
   - Distributed Spark SQL aggregated metrics table: total sales, daily demand mean, demand volatility (standard deviation), and single-day peak volume.
   - Longitudinal monthly sales volume charts spanning 2014 through 2019.

---

## 🧪 Comprehensive End-to-End Verification Suite

To verify system integrity without running the UI, run the automated verification script:
```bash
python verify_project_end_to_end.py
```

### Verification Checks & Results (100% Passed):
- `[CHECK 1/6] PySpark Data Pipeline`: Validates 16,848 rows in `daily_sales.parquet`, 8 rows in `medicine_summary.parquet`, 560 rows in `monthly_trends.parquet`. Verifies 0 duplicate keys, 0 negative values, 0 null dates, and `0.000000` volume discrepancy against raw CSV.
- `[CHECK 2/6] Demand Forecasting`: Confirms 56 rows in `forecast_7days.parquet` (all non-negative). Verifies that Random Forest outperformed the 7-day Moving Average baseline on all 8/8 medicines in the 60-day chronological holdout test.
- `[CHECK 3/6] Inventory Stockout Risk`: Confirms formula compliance: $\text{Inventory Position} = \text{Current Stock} + \text{Pending Order}$, $\text{ROP} = \text{Lead-Time Demand} + \text{Safety Stock}$.
- `[CHECK 4/6] Batch Expiry Risk (FEFO)`: Validates that batches with $\text{days\_until\_expiry} \le 0$ are classified as `EXPIRED` with strict disposal directives. Validates that $\text{Value at Risk} = \text{Unsold Units at Risk} \times \text{Unit Cost}$ holds exactly to the cent.
- `[CHECK 5/6] Smart Reordering Engine`: Validates non-negative order quantities, confirms order quantity is 0 for non-urgent medicines, and validates that pending orders are properly deducted (e.g., `N02BE` ordered 388 units instead of 438 because 50 units were in transit).
- `[CHECK 6/6] Streamlit Data Loader`: Ingests all 8 output tables into memory (17,512 total records) without errors.

---

## 📥 Sales Data Ingestion & Import Engine (Add Medicine Feature)

PHARMORA includes a production-grade, non-destructive sales ingestion engine allowing pharmacists and supply chain managers to import supplementary sales records from CSV files directly via the Streamlit dashboard or programmatic Python modules.

### 1. Supported CSV Schemas

#### Standard Long Format (Recommended)
| Column Name | Data Type | Requirement | Description | Example |
|:---|:---|:---|:---|:---|
| `date` | String / Date | **Required** | Transaction date in ISO (`YYYY-MM-DD`), US (`M/D/YYYY`), or EU (`D/M/YYYY`) format | `2020-01-15` or `1/15/2020` |
| `medicine_id` | String | **Required** | Active ATC therapeutic code or pharmaceutical identifier | `J01CA`, `N02BE` |
| `units_sold` | Float / Int | **Required** | Non-negative sales volume ($\ge 0$). Nulls or negative values rejected. | `14.5` |
| `transaction_id`| String | *Optional* | Unique identifier for granular multi-order tracking | `TX-20200115-001` |

#### Wide ATC Matrix Format (Legacy & Baseline Compatible)
Files formatted identically to `data/salesdaily.csv` (e.g., containing `datum`, `M01AB`, `J01CA`...) are automatically detected, sanitized, and unpivoted to normalized long format.

### 2. Validation & Quality Rules
Before accepting any file for ingestion, PHARMORA executes comprehensive row-by-row data quality audits:
- **Schema Conformity**: Ensures all required column headers are present.
- **Date Validity**: Multi-format date parser validates that every record contains a legitimate calendar date.
- **Non-Negative Sales Volume**: Strict verification that $\text{units\_sold} \ge 0.0$. Negative quantities are rejected with line-numbered error flags.
- **Valid Medicine Identifier**: Non-empty, alphanumeric strings; nulls and whitespace-only entries are rejected.
- **Error Register**: In the Streamlit UI, any invalid rows are presented in a transparent diagnostic table displaying the exact row number, field name, rejected value, and descriptive error reason.

### 3. Deduplication Policy & Idempotent Re-Uploads
To prevent duplicate facts when the same file is uploaded repeatedly or overlapping transaction batches are ingested:
- **Composite Key Matching**: If `transaction_id` is omitted, deduplication checks composite key `(date, medicine_id)`. If an exact date-medicine fact already exists in `daily_sales.parquet` or `data/imported_sales.csv`, it is skipped.
- **Transaction ID Tracking**: If `transaction_id` is provided, deduplication is enforced strictly by transaction ID, allowing multiple legitimate distinct sales transactions for the same medicine on the same day.
- **Idempotency**: Re-uploading an identical CSV produces 0 duplicate records and 0 ingested facts, protecting database integrity.

### 4. Non-Destructive Storage Architecture
- **Baseline Inviolability**: The original baseline dataset (`data/salesdaily.csv`) is **never modified or overwritten**.
- **Persistent Staging**: All validated records are appended to `data/imported_sales.csv` with audit metadata:
  - `imported_at`: ISO timestamp of ingestion.
  - `source_file`: Original uploaded filename.
- **Incremental Parquet Synchronization**: `persist_and_update_pipeline()` updates `processed/daily_sales.parquet`, recalculates summary statistics in `processed/medicine_summary.parquet`, and updates longitudinal trends in `processed/monthly_trends.parquet`.
- **PySpark Integration**: `src/data_pipeline.py` automatically detects and unions `data/imported_sales.csv` during batch PySpark rebuilds.

### 5. Downstream Analytics & Zero-Fabrication Guarantee
- **Dynamic Selector**: Newly imported medicines immediately appear in the Streamlit medicine selector.
- **Historical Sales Slices**: Daily sales charts and moving averages update reactively.
- **Demand Forecasting**:
  - If a medicine has $< 60$ historical daily records: Displays a prominent notice: *"⚠️ Insufficient history for machine learning forecasting (< 60 chronological days)"* and provides an explicitly labelled fallback based on historical daily mean demand ($\mu$).
  - **Zero Fabrication**: No synthetic holdout accuracy metrics or fabricated forecast curves are assigned.
- **Warehouse Inventory & Batch Tracking**:
  - Physical stock, batch numbers, and expiration dates are **never fabricated**.
  - If a new medicine lacks entries in `data/inventory.csv` or `data/batches.csv`, the dashboard displays a clear configuration guide explaining how to register warehouse parameters.

### 6. Automated Feature Verification
Execute the dedicated test suite verifying all import capabilities:
```bash
python test_import_sales_feature.py
```

---

## 💻 Complete Laptop Demonstration Run Sequence

To demonstrate the complete PHARMORA project from scratch on a laptop, execute the following commands in sequence:

```powershell
# Step 1: Open project directory
cd c:\Pharmora

# Step 2: Run the PySpark Big Data Pipeline (Phase 1)
python -m src.data_pipeline

# Step 3: Train Machine Learning Forecasters & Generate 7-Day Forecast (Phase 2)
python -m src.forecasting

# Step 4: Run the Inventory Risk Engine for Stockout & FEFO Expiry (Phase 3)
python -m src.inventory

# Step 5: Run the Smart Reordering Engine for Prescriptive Procurement (Phase 4)
python -m src.reordering

# Step 6: Execute the Automated Integration & Verification Suite
python verify_project_end_to_end.py

# Step 7: Launch the Streamlit Interactive Intelligence Dashboard (Phase 5)
streamlit run app.py
```

---

## 📐 Algorithmic Framework Summary

| Module | Core Algorithms & Methods | Inputs | Outputs |
|:---|:---|:---|:---|
| **Phase 1: PySpark Pipeline** | PySpark DataFrame API, schema validation, wide-to-long unpivot transformation, Spark SQL Window functions | `data/salesdaily.csv` (2,106 rows) | `processed/daily_sales.parquet`, `medicine_summary.parquet`, `monthly_trends.parquet` |
| **Phase 2: Demand Forecasting** | Scikit-learn `RandomForestRegressor`, chronological train/test split (no shuffle), multi-step recursive forecasting, lag features ($t-1, t-2, t-3, t-7, t-14, t-21, t-28$), 7-day & 14-day rolling statistics, calendar indicators | `processed/daily_sales.parquet` | `processed/forecast_7days.parquet`, `forecast_evaluations.parquet`, 8 `.joblib` model files |
| **Phase 3: Inventory Risk Engine** | Net Inventory Position modeling, Statistical Reorder Point ($\text{ROP} = \mu_L + z \sigma_L$), Zero-division safe Days of Stock Cover, FEFO priority queue with multi-batch cumulative demand absorption | `data/inventory.csv`, `data/batches.csv`, `processed/forecast_7days.parquet` | `processed/inventory_stockout_risk.parquet`, `batch_expiry_risk.parquet` |
| **Phase 4: Smart Reordering** | Periodic review $(R, S)$ order-up-to policy, Lead-time + review period demand projection, In-flight pending order netting, Discrete ceiling quantization, Multi-tier urgency horizon classification, Healthcare expiry safety checks | `data/inventory.csv`, `processed/forecast_7days.parquet`, `batch_expiry_risk.parquet` | `processed/reorder_recommendations.parquet`, `reorder_recommendations.csv` |
| **Phase 5: Intelligence Dashboard** | Streamlit component architecture, Plotly interactive visualizations, on-the-fly model backtesting, dynamic what-if simulation, cross-module Parquet ingestion | All `processed/*.parquet` artifacts | Interactive web application at `http://localhost:8501` |

---

## 📋 Data Sources & Synthetic Data Labelling

1. **Transactional Sales Dataset (`data/salesdaily.csv`)**:
   - **Type**: Real-world pharmaceutical retail data.
   - **Source**: Open-access pharmaceutical sales time-series (2014-2019, 2,106 daily records).
   - **ATC Categories**: 8 standardized therapeutic classes (`M01AB`, `M01AE`, `N02BA`, `N02BE`, `N05B`, `N05C`, `R03`, `R06`).
2. **Warehouse Inventory Dataset (`data/inventory.csv`)**:
   - **Type**: **Synthetic Demonstration Dataset**.
   - **Rationale**: Real hospital/pharmacy stock levels and supplier contract lead times are confidential. Realistic parameters (current stock, lead times 5-10 days, safety stock, pending orders, and unit costs) were synthesized for the 8 active ATC categories to demonstrate stockout detection and ROP calculations.
3. **Batch Expiry Dataset (`data/batches.csv`)**:
   - **Type**: **Synthetic Demonstration Dataset**.
   - **Rationale**: Created to demonstrate First-Expired, First-Out (FEFO) dispensing queues and multi-batch shelf-life tracking. Contains 16 batches across the 8 ATC categories with varied manufacturing and expiration dates (including past-dated batches to demonstrate safety quarantine protocols).

---

## 📈 Summary of Actual Evaluation Results

### Demand Forecasting (60-Day Chronological Holdout)
- **Baseline (7-day MA)**: Overall MAE = 2.917 units | Overall RMSE = 3.841 units
- **Random Forest**: Overall MAE = **2.648 units** | Overall RMSE = **3.563 units**
- **Performance**: Random Forest outperformed the baseline on **all 8 out of 8 medicines** with an average MAE error reduction of **9.24%** (up to 14.12% improvement on `R06`).

### Inventory Stockout Audit
- **Critical Stockout Risk**: 2 medicines (`M01AB` Diclofenac with 3.6 days cover; `N02BE` Paracetamol with 3.7 days cover).
- **Reorder Warnings**: 2 medicines (`M01AE` Ibuprofen with 6.1 days cover; `R03` Asthma medication with 7.2 days cover).
- **Healthy Buffer**: 3 medicines (`N02BA` Aspirin, `N05B` Anxiolytics, `R06` Antihistamines).
- **Overstocked**: 1 medicine (`N05C` Sedatives with 61.9 days cover).

### Batch Expiry Audit (FEFO Allocation)
- **Expired Batches Quarantined**: 5 batches (80.0 total units; financial loss: **₹898.75**). Action: Strict medical quarantine; DO NOT SELL.
- **Near-Expiry Risk Batches**: 4 batches (35.1 units at risk; capital exposed: **₹544.45**). Action: Expedited FEFO dispensing or vendor return.
- **Healthy Batches**: 7 batches (100% of units absorbed by projected demand).
- **Total Financial Capital at Risk**: **₹1,443.20**.

### Prescriptive Reordering
- **Urgent Replenishment Required**: 4 medicines (`M01AB`: 64 units, `M01AE`: 33 units, `N02BE`: 388 units, `R03`: 65 units).
- **Total Reorder Volume**: **550 discrete units** across 4 urgent categories.
- **Upcoming Reorder Horizon**: 3 medicines (`N02BA`, `N05B`, `R06`) approaching ROP within 3.4 to 6.7 days.
- **Reorder Not Required**: 1 medicine (`N05C`).

---

## ⚠️ Known Project Limitations & Assumptions

1. **Synthetic Inventory & Batch Master**: Warehouse stock and batch tracking records are synthetic demonstrations mapped to real ATC demand curves. In production, these tables would be synchronized via live ERP connectors (e.g., SAP, Oracle, Epic Systems).
2. **Deterministic Supplier Lead Times**: Lead times are currently treated as deterministic ($L$ days). In stochastic supply chain environments, lead time variance should be integrated into safety stock calculations ($z \times \sqrt{L \sigma_d^2 + d^2 \sigma_L^2}$).
3. **Discrete Unit Ordering / Minimum Order Quantities (MOQ)**: Recommended order quantities are rounded to discrete units. Suppliers with minimum packaging constraints (e.g., cases of 24 or 50) require MOQ ceiling rules in the procurement catalog.
4. **Single Supplier per Medicine Category**: The reordering engine assumes a single primary distributor per ATC category. Multi-sourcing allocation models can be added for enterprise resilience.
5. **No Cross-Price Elasticity or Drug Substitution**: The models treat the 8 ATC therapeutic categories independently without modeling intra-category drug substitutions during severe shortages.
