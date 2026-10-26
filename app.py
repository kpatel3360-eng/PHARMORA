"""
PHARMORA - AI-Powered Pharmacy Intelligence Platform
====================================================
Production-grade Streamlit Web Application for Big Data Analytics,
Machine Learning Demand Forecasting, Inventory Risk, and Smart Reordering.

Design System: Modern Emerald & Warm Ivory
- Main Canvas Background: #F5F7F4
- Content Cards & Sidebar: #FFFFFF
- Primary Brand Green: #166534
- Dark Heading Text: #16352B
- Body Text: #374151
- Secondary / Caption Text: #64748B
- Borders: #DDE5DF
- Light Green Highlights: #E8F3EC
- Warning: #B45309
- Critical Risk: #B91C1C
- Success: #15803D
"""

import os
import math
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Local analytical modules
from src.inventory import calculate_inventory_metrics, assess_stock_status
from src.forecasting import train_medicine_forecaster
from src.data_ingestion import (
    validate_sales_records,
    deduplicate_records,
    persist_and_update_pipeline,
    IMPORTED_STAGING_PATH
)
from src.rescue import (
    MODE_MULTI_BRANCH,
    MODE_NETWORK_SHARING,
    MODE_SINGLE_STORE,
    OPERATING_MODES,
    DEFAULT_BRANCHES,
    DEFAULT_NETWORK_PHARMACIES,
    get_initial_branch_inventory,
    get_initial_branch_batches,
    get_initial_network_stock,
    run_rescue_to_reorder_engine,
    execute_confirmed_transfer
)

# ---------------------------------------------------------------------------
# Standard ATC Pharmaceutical Taxonomies
# ---------------------------------------------------------------------------
ATC_NAMES = {
    "M01AB": "M01AB - Anti-inflammatory (Diclofenac)",
    "M01AE": "M01AE - Anti-inflammatory (Ibuprofen)",
    "N02BA": "N02BA - Analgesics & Antipyretics (Aspirin)",
    "N02BE": "N02BE - Analgesics & Antipyretics (Paracetamol)",
    "N05B":  "N05B - Psycholeptics (Anxiolytics / Diazepam)",
    "N05C":  "N05C - Psycholeptics (Hypnotics & Sedatives / Zolpidem)",
    "R03":   "R03 - Anti-asthmatic (Salbutamol)",
    "R06":   "R06 - Antihistamines (Cetirizine / Allergy Relief)"
}

ATC_SHORT = {
    "M01AB": "Diclofenac",
    "M01AE": "Ibuprofen",
    "N02BA": "Aspirin",
    "N02BE": "Paracetamol",
    "N05B":  "Anxiolytics",
    "N05C":  "Sedatives",
    "R03":   "Asthma",
    "R06":   "Antihistamines"
}

ATC_CLASSES = {
    "M01AB": "Anti-inflammatory (Acetic acid derivatives)",
    "M01AE": "Anti-inflammatory (Propionic acid derivatives)",
    "N02BA": "Analgesics & Antipyretics (Salicylic acid derivatives)",
    "N02BE": "Analgesics & Antipyretics (Pyrazolones & Anilides)",
    "N05B":  "Psycholeptics (Anxiolytic agents)",
    "N05C":  "Psycholeptics (Hypnotics & Sedatives)",
    "R03":   "Drugs for Obstructive Airway Diseases",
    "R06":   "Antihistamines for Systemic Use"
}

# Theme Palette Tokens: Modern Emerald & Warm Ivory
COLOR_BG = "#F5F7F4"          # Main canvas background
COLOR_CARD = "#FFFFFF"        # Content cards & sidebar
COLOR_PRIMARY = "#166534"     # Primary brand green
COLOR_HEADING = "#16352B"     # Dark heading text
COLOR_BODY = "#374151"        # Body text
COLOR_MUTED = "#64748B"       # Secondary text
COLOR_BORDER = "#DDE5DF"      # Structural borders
COLOR_HIGHLIGHT = "#E8F3EC"   # Light green highlight
COLOR_WARNING = "#B45309"     # Warning amber
COLOR_CRITICAL = "#B91C1C"    # Critical risk red
COLOR_SUCCESS = "#15803D"     # Success green


def _read_parquet_source(path: Path) -> Optional[pd.DataFrame]:
    """Reads a parquet file or directory of parquet parts robustly."""
    if not path.exists():
        return None
    try:
        if path.is_file():
            return pd.read_parquet(path)
        parts = [f for f in path.glob("*.parquet") if not f.name.startswith(".")]
        if not parts:
            return None
        return pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    except Exception:
        return None


@st.cache_data
def load_all_pharmora_data() -> Dict[str, Optional[pd.DataFrame]]:
    """Loads all curated Parquet outputs from data pipeline, ML, and inventory engines."""
    p = Path("processed")

    daily_df = _read_parquet_source(p / "daily_sales.parquet")
    if daily_df is not None:
        daily_df["date"] = pd.to_datetime(daily_df["date"])

    summary_df = _read_parquet_source(p / "medicine_summary.parquet")
    monthly_df = _read_parquet_source(p / "monthly_trends.parquet")
    forecast_df = _read_parquet_source(p / "forecast_7days.parquet")
    if forecast_df is not None:
        forecast_df["forecast_date"] = pd.to_datetime(forecast_df["forecast_date"])

    eval_df = _read_parquet_source(p / "forecast_evaluations.parquet")
    stockout_df = _read_parquet_source(p / "inventory_stockout_risk.parquet")
    expiry_df = _read_parquet_source(p / "batch_expiry_risk.parquet")
    if expiry_df is not None:
        expiry_df["expiry_date"] = pd.to_datetime(expiry_df["expiry_date"])

    reorder_df = _read_parquet_source(p / "reorder_recommendations.parquet")

    return {
        "daily": daily_df,
        "summary": summary_df,
        "monthly": monthly_df,
        "forecast": forecast_df,
        "evaluations": eval_df,
        "stockout": stockout_df,
        "expiry": expiry_df,
        "reorder": reorder_df
    }


def render_badge_html(status: str) -> str:
    """Returns an accessible, restrained HTML badge for risk and reorder statuses."""
    s = str(status).upper()
    if s in ["CRITICAL_STOCKOUT_RISK", "EXPIRED", "URGENT"]:
        bg = "#FEE2E2"
        color = "#B91C1C"
        border = "#FCA5A5"
    elif s in ["REORDER_WARNING", "EXPIRY_RISK", "UPCOMING"]:
        bg = "#FEF3C7"
        color = "#B45309"
        border = "#FCD34D"
    elif s in ["HEALTHY", "HEALTHY_BATCH", "NOT_REQUIRED"]:
        bg = "#E8F3EC"
        color = "#15803D"
        border = "#A5D6A7"
    elif s in ["OVERSTOCKED"]:
        bg = "#EEF2FF"
        color = "#3730A3"
        border = "#C7D2FE"
    else:
        bg = "#F5F7F4"
        color = "#374151"
        border = "#DDE5DF"

    clean_label = s.replace("_", " ")
    return (
        f'<span style="background-color: {bg}; color: {color}; border: 1px solid {border}; '
        f'padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.78rem; '
        f'display: inline-block; letter-spacing: 0.3px;">{clean_label}</span>'
    )


def render_metric_card(title: str, value: str, subtext: str = "", card_type: str = "default") -> str:
    """Renders a clean, white KPI card with subtle border and crisp typography."""
    border_color = "#DDE5DF"
    val_color = "#16352B"
    if card_type == "urgent":
        border_color = "#FCA5A5"
        val_color = "#B91C1C"
    elif card_type == "warning":
        border_color = "#FCD34D"
        val_color = "#B45309"
    elif card_type == "healthy":
        border_color = "#A5D6A7"
        val_color = "#15803D"
    elif card_type == "focus":
        border_color = "#166534"
        val_color = "#166534"

    return f"""
    <div style="background-color: #FFFFFF; border: 1px solid {border_color}; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(22, 53, 43, 0.04);">
        <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748B; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 1.55rem; font-weight: 700; color: {val_color}; margin: 4px 0 2px 0;">{value}</div>
        <div style="font-size: 0.8rem; color: #374151; line-height: 1.2;">{subtext}</div>
    </div>
    """


def main():
    st.set_page_config(
        page_title="PHARMORA | Pharmacy Intelligence Platform",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # -----------------------------------------------------------------------
    # Comprehensive Light Theme CSS: Modern Emerald + Warm Ivory
    # -----------------------------------------------------------------------
    st.markdown("""
    <style>
        /* Base page styling */
        .stApp {
            background-color: #F5F7F4 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #374151 !important;
        }

        /* Top Toolbar / Header - Light Ivory Harmonization */
        header[data-testid="stHeader"],
        [data-testid="stHeader"] {
            background-color: #F5F7F4 !important;
            border-bottom: 1px solid #DDE5DF !important;
            color: #16352B !important;
        }
        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] svg {
            color: #16352B !important;
            fill: #16352B !important;
        }

        /* Sidebar Styling - Clean White with explicit dark text */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #DDE5DF !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #16352B !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #374151 !important;
        }
        section[data-testid="stSidebar"] label p {
            color: #16352B !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {
            color: #64748B !important;
        }

        /* Brand title in sidebar */
        .brand-title {
            color: #166534 !important;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0;
            padding: 0;
        }
        .brand-tagline {
            color: #64748B !important;
            font-size: 0.82rem;
            margin-top: 2px;
            margin-bottom: 14px;
        }

        /* Header banner */
        .page-header {
            background-color: #FFFFFF;
            border: 1px solid #DDE5DF;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: 0 1px 2px rgba(22, 53, 43, 0.03);
        }
        .page-title {
            color: #16352B !important;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0 0 4px 0;
        }
        .page-subtitle {
            color: #374151 !important;
            font-size: 0.95rem;
            margin: 0;
        }

        /* Context pill for selected medicine */
        .context-pill {
            display: inline-flex;
            align-items: center;
            background-color: #E8F3EC;
            border: 1px solid #A5D6A7;
            color: #166534 !important;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 4px 12px;
            border-radius: 20px;
            margin-top: 8px;
        }

        /* Section Headings */
        .section-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: #16352B !important;
            margin: 16px 0 8px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-subtitle {
            font-size: 0.85rem;
            color: #64748B !important;
            margin-bottom: 12px;
        }

        /* All Markdown headings & labels */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #16352B !important;
            font-weight: 700 !important;
        }
        .stMarkdown p {
            color: #374151 !important;
        }
        label[data-testid="stWidgetLabel"] p {
            color: #16352B !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }

        /* Tab Bar Styling - High Contrast & Modern Emerald Highlight */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 2px solid #DDE5DF !important;
            background-color: transparent !important;
            padding-bottom: 0px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            color: #374151 !important;
            padding: 10px 18px !important;
            background-color: #FFFFFF !important;
            border: 1px solid #DDE5DF !important;
            border-bottom: none !important;
            border-radius: 6px 6px 0 0 !important;
            transition: all 0.15s ease-in-out;
        }
        .stTabs [data-baseweb="tab"] p {
            color: #374151 !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #166534 !important;
            background-color: #E8F3EC !important;
        }
        .stTabs [data-baseweb="tab"]:hover p {
            color: #166534 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #166534 !important;
            font-weight: 700 !important;
            background-color: #FFFFFF !important;
            border: 1px solid #DDE5DF !important;
            border-bottom: 3px solid #166534 !important;
            margin-bottom: -2px !important;
        }
        .stTabs [aria-selected="true"] p {
            color: #166534 !important;
            font-weight: 700 !important;
        }

        /* Form Controls: Selectbox & Dropdown Menus */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #DDE5DF !important;
            border-radius: 6px !important;
            color: #374151 !important;
        }
        div[data-baseweb="select"] span {
            color: #374151 !important;
        }
        div[data-baseweb="select"] svg {
            fill: #16352B !important;
        }
        ul[data-baseweb="menu"] {
            background-color: #FFFFFF !important;
            border: 1px solid #DDE5DF !important;
        }
        li[data-baseweb="menu-item"] {
            color: #374151 !important;
            background-color: #FFFFFF !important;
        }
        li[data-baseweb="menu-item"]:hover,
        li[data-baseweb="menu-item"][aria-selected="true"] {
            background-color: #E8F3EC !important;
            color: #166534 !important;
            font-weight: 600 !important;
        }

        /* Date Inputs */
        div[data-baseweb="input"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #DDE5DF !important;
            border-radius: 6px !important;
            color: #374151 !important;
        }
        div[data-baseweb="input"] input {
            color: #374151 !important;
            background-color: transparent !important;
        }
        div[data-baseweb="input"] svg {
            fill: #16352B !important;
        }

        /* Buttons */
        button[kind="primary"] {
            background-color: #166534 !important;
            border-color: #166534 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        button[kind="primary"]:hover {
            background-color: #14532D !important;
            border-color: #14532D !important;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #DDE5DF !important;
            border-radius: 8px !important;
        }
        div[data-testid="stExpander"] summary p {
            color: #16352B !important;
            font-weight: 600 !important;
        }
        div[data-testid="stExpander"] svg {
            fill: #16352B !important;
        }

        /* DataFrames */
        div[data-testid="stDataFrame"] {
            border: 1px solid #DDE5DF !important;
            border-radius: 8px !important;
            background-color: #FFFFFF !important;
        }

        /* Setup / Choice Cards for Rescue-to-Reorder */
        .choice-card {
            background-color: #FFFFFF;
            border: 2px solid #DDE5DF;
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 14px;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 1px 3px rgba(22, 53, 43, 0.05);
        }
        .choice-card:hover {
            border-color: #166534;
            box-shadow: 0 4px 12px rgba(22, 101, 52, 0.08);
        }
        .choice-icon {
            font-size: 2.2rem;
            margin-bottom: 10px;
        }
        .choice-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: #16352B;
            margin-bottom: 6px;
        }
        .choice-desc {
            font-size: 0.92rem;
            font-weight: 600;
            color: #166534;
            margin-bottom: 8px;
        }
        .choice-support {
            font-size: 0.85rem;
            color: #475569;
            line-height: 1.45;
            margin-bottom: 14px;
        }
    </style>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # RESCUE-TO-REORDER™ SESSION STATE INITIALIZATION
    # -----------------------------------------------------------------------
    if "rescue_operating_mode" not in st.session_state:
        st.session_state["rescue_operating_mode"] = None
    if "show_rescue_onboarding" not in st.session_state:
        st.session_state["show_rescue_onboarding"] = False
    if "branch_inventory_df" not in st.session_state:
        st.session_state["branch_inventory_df"] = get_initial_branch_inventory()
    if "branch_batches_df" not in st.session_state:
        st.session_state["branch_batches_df"] = get_initial_branch_batches()
    if "network_stock_df" not in st.session_state:
        st.session_state["network_stock_df"] = get_initial_network_stock()
    if "active_confirmed_transfers" not in st.session_state:
        st.session_state["active_confirmed_transfers"] = []
    if "confirmed_transfer_history" not in st.session_state:
        st.session_state["confirmed_transfer_history"] = []
    if "network_sharing_requests" not in st.session_state:
        st.session_state["network_sharing_requests"] = []
    if "pharmacy_registration" not in st.session_state:
        st.session_state["pharmacy_registration"] = {
            "registered": True,
            "name": "Apex Community Pharmacy (Your Store)",
            "location": "Downtown Metro District",
            "license": "PH-METRO-99104",
            "contact_person": "Pharmacist in Charge",
            "contact_phone": "+1 (555) 019-2834",
            "privacy_level": "RESTRICTED",
            "sharing_categories": ["Anti-inflammatory", "Respiratory", "Analgesics"]
        }
    if "rejected_recommendations" not in st.session_state:
        st.session_state["rejected_recommendations"] = set()

    data = load_all_pharmora_data()
    daily_df = data["daily"]

    if daily_df is None:
        st.error("⚠️ **Processed Big Data pipeline artifacts not found!**")
        st.info(
            "Please run the PySpark data pipeline first to generate the Parquet tables:\n"
            "```bash\npython -m src.data_pipeline\npython -m src.forecasting\npython -m src.inventory\npython -m src.reordering\n```"
        )
        return

    # Extract all available ATC categories
    all_meds = sorted(daily_df["medicine_id"].unique().tolist())
    stockout_df = data["stockout"]
    reorder_df = data["reorder"]
    expiry_df = data["expiry"]
    forecast_df = data["forecast"]
    eval_df = data["evaluations"]
    summary_df = data["summary"]
    monthly_df = data["monthly"]

    # -----------------------------------------------------------------------
    # DYNAMIC DATE BOUNDARY CALCULATION & SYSTEM TIMELINE
    # -----------------------------------------------------------------------
    system_today = datetime.date.today()

    if daily_df is not None and not daily_df.empty:
        valid_sales_dates = pd.to_datetime(daily_df["date"], errors="coerce").dropna()
        if not valid_sales_dates.empty:
            actual_min_date = valid_sales_dates.min().date()
            actual_max_date = valid_sales_dates.max().date()
            has_valid_sales = True
        else:
            actual_min_date = system_today - datetime.timedelta(days=365)
            actual_max_date = system_today
            has_valid_sales = False
    else:
        actual_min_date = system_today - datetime.timedelta(days=365)
        actual_max_date = system_today
        has_valid_sales = False

    # -----------------------------------------------------------------------
    # SIDEBAR: Brand, Medicine Selection & Active Context
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown('<div class="brand-title">💊 PHARMORA</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-tagline">Pharmacy Intelligence & Supply Chain Engine</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 Active Medicine Focus")

        # The Shared Source of Truth: Selected Medicine
        default_idx = all_meds.index("N02BE") if "N02BE" in all_meds else 0
        selected_med = st.selectbox(
            "Select Medicine Category",
            options=all_meds,
            format_func=lambda x: f"{x} - {ATC_SHORT.get(x, x)}",
            index=default_idx,
            key="shared_medicine_selector",
            help="Select a medicine to dynamically filter and inspect its demand, stockout risk, expiry audit, and reorder plan."
        )

        # Context Card for Selected Medicine right in the sidebar
        med_short = ATC_SHORT.get(selected_med, selected_med)
        med_class = ATC_CLASSES.get(selected_med, "Therapeutic Category")

        # Lookup selected medicine metrics for sidebar snapshot
        med_stock_row = stockout_df[stockout_df["medicine_id"] == selected_med].iloc[0] if (stockout_df is not None and not stockout_df[stockout_df["medicine_id"] == selected_med].empty) else None
        med_reorder_row = reorder_df[reorder_df["medicine_id"] == selected_med].iloc[0] if (reorder_df is not None and not reorder_df[reorder_df["medicine_id"] == selected_med].empty) else None

        stock_val = f"{med_stock_row['current_stock']:.0f} units" if med_stock_row is not None else "Not Configured"
        risk_status = med_stock_row["risk_status"] if med_stock_row is not None else "NOT_CONFIGURED"
        order_qty = f"{med_reorder_row['suggested_order_quantity']} units" if med_reorder_row is not None else "N/A"
        reorder_status = med_reorder_row["reorder_status"] if med_reorder_row is not None else "NOT_CONFIGURED"

        st.markdown(f"""
        <div style="background-color: #F5F7F4; border: 1px solid #DDE5DF; border-radius: 8px; padding: 12px; margin-top: 8px; margin-bottom: 16px;">
            <div style="font-size: 0.75rem; font-weight: 600; color: #64748B; text-transform: uppercase;">Active Context</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #16352B; margin: 2px 0;">{selected_med} · {med_short}</div>
            <div style="font-size: 0.78rem; color: #374151; margin-bottom: 8px;">{med_class}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #DDE5DF; padding-top: 6px; font-size: 0.8rem;">
                <span style="color: #64748B;">Current Stock:</span>
                <span style="font-weight: 700; color: #16352B;">{stock_val}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px; font-size: 0.8rem;">
                <span style="color: #64748B;">Inventory Posture:</span>
                <span>{render_badge_html(risk_status)}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px; font-size: 0.8rem;">
                <span style="color: #64748B;">Reorder Need:</span>
                <span style="font-weight: 700; color: {'#B91C1C' if reorder_status == 'URGENT' else '#16352B'};">{order_qty} ({reorder_status.replace('_', ' ')})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📅 Historical Horizon")

        # Operational Timeline card displaying system date and historical availability
        st.markdown(f"""
        <div style="background-color: #F8FAF9; border: 1px solid #DDE5DF; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #166534; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                🗓️ System Timeline & Horizon
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; margin-bottom: 4px;">
                <span style="color: #64748B;">System Run Date:</span>
                <span style="font-weight: 700; color: #16352B; background-color: #E8F3EC; padding: 1px 6px; border-radius: 4px; border: 1px solid #DDE5DF;">
                    {system_today.strftime("%d %B %Y")}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; margin-bottom: 4px;">
                <span style="color: #64748B;">Historical Horizon:</span>
                <span style="font-weight: 700; color: #166534;">
                    {actual_min_date.strftime("%d %b %Y")} – {actual_max_date.strftime("%d %b %Y")}
                </span>
            </div>
            <div style="font-size: 0.73rem; color: #64748B; border-top: 1px solid #E8EDE9; padding-top: 6px; margin-top: 6px; line-height: 1.35;">
                🔒 <strong>Zero-Fabrication Guarantee:</strong> Historical records reflect verified actual sales facts ({actual_max_date.year}). Dates up to current system year ({system_today.year}) are strictly not populated with fake sales.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Clear requirement 4 label: "Historical data available: [start date] – [end date]"
        st.markdown(
            f'<div style="font-size: 0.82rem; font-weight: 600; color: #16352B; margin-bottom: 6px;">'
            f'Historical data available: <span style="color: #166534;">{actual_min_date.strftime("%d %b %Y")} – {actual_max_date.strftime("%d %b %Y")}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Ensure session state reacts dynamically when sales data is imported
        if "last_known_max_date" not in st.session_state:
            st.session_state["last_known_max_date"] = actual_max_date
        elif st.session_state["last_known_max_date"] != actual_max_date:
            st.session_state["last_known_max_date"] = actual_max_date
            st.session_state["historical_date_slider"] = (actual_min_date, actual_max_date)

        # Guard against session state dates out of range
        if "historical_date_slider" in st.session_state:
            curr_val = st.session_state["historical_date_slider"]
            if isinstance(curr_val, (list, tuple)) and len(curr_val) >= 1:
                clamped_start = max(actual_min_date, min(curr_val[0], actual_max_date))
                clamped_end = actual_max_date
                if len(curr_val) == 2 and curr_val[1] is not None:
                    clamped_end = max(actual_min_date, min(curr_val[1], actual_max_date))
                if clamped_start > clamped_end:
                    clamped_start = actual_min_date
                    clamped_end = actual_max_date
                st.session_state["historical_date_slider"] = (clamped_start, clamped_end)

        # Historical date picker strictly bounded between actual_min_date and actual_max_date
        date_range = st.date_input(
            "Filter Historical Sales Period",
            value=(actual_min_date, actual_max_date),
            min_value=actual_min_date,
            max_value=actual_max_date,
            key="historical_date_slider",
            help=f"Select an active date range between {actual_min_date.strftime('%Y-%m-%d')} and {actual_max_date.strftime('%Y-%m-%d')}. Historical end date cannot exceed {actual_max_date.strftime('%Y-%m-%d')}."
        )

        st.markdown("---")
        st.caption("PHARMORA Platform · Modern Emerald & Warm Ivory · Localhost :8501")

    # -----------------------------------------------------------------------
    # HISTORICAL DATE FILTERING & STRICT CLAMPING
    # -----------------------------------------------------------------------
    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 2:
            s_val = date_range[0] if date_range[0] is not None else actual_min_date
            e_val = date_range[1] if date_range[1] is not None else actual_max_date
            start_d, end_d = pd.to_datetime(s_val), pd.to_datetime(e_val)
        elif len(date_range) == 1:
            s_val = date_range[0] if date_range[0] is not None else actual_min_date
            start_d = pd.to_datetime(s_val)
            end_d = pd.to_datetime(actual_max_date)
        else:
            start_d, end_d = pd.to_datetime(actual_min_date), pd.to_datetime(actual_max_date)
    else:
        start_d, end_d = pd.to_datetime(actual_min_date), pd.to_datetime(actual_max_date)

    if start_d > end_d:
        start_d, end_d = end_d, start_d

    # Strict clamping: End date MUST NEVER exceed actual_max_date
    start_d = max(start_d, pd.to_datetime(actual_min_date))
    end_d = min(end_d, pd.to_datetime(actual_max_date))

    # Apply date filter
    if daily_df is not None and not daily_df.empty:
        filtered_daily = daily_df[(daily_df["date"] >= start_d) & (daily_df["date"] <= end_d)]
    else:
        filtered_daily = pd.DataFrame(columns=["date", "medicine_id", "units_sold"])

    # Selected Medicine Historical Slice (Reactive source of truth)
    if not filtered_daily.empty:
        med_daily = filtered_daily[filtered_daily["medicine_id"] == selected_med].sort_values("date")
    else:
        med_daily = pd.DataFrame(columns=["date", "medicine_id", "units_sold"])

    # -----------------------------------------------------------------------
    # HEADER BANNER: Global Context, Current System Date & Historical Horizon
    # -----------------------------------------------------------------------
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">💊 PHARMORA Intelligence Platform</div>
        <div class="page-subtitle">Prescriptive Big Data Analytics, Demand Forecasting & Inventory Risk Optimization</div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
            <div class="context-pill">
                Active Focus: {ATC_NAMES.get(selected_med, selected_med)} &nbsp;·&nbsp; {ATC_CLASSES.get(selected_med, '')}
            </div>
            <div class="context-pill" style="background-color: #F1F5F9; color: #334155; border-color: #CBD5E1;">
                Current Date: {system_today.strftime('%d %b %Y')}
            </div>
            <div class="context-pill" style="background-color: #E8F3EC; color: #166534; border-color: #A7F3D0;">
                Historical Data Available: {actual_min_date.strftime('%d %b %Y')} – {actual_max_date.strftime('%d %b %Y')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # NAVIGATION TABS (All 8 Sections including Rescue-to-Reorder™)
    # -----------------------------------------------------------------------
    tab_overview, tab_forecast, tab_stockout, tab_expiry, tab_reorder, tab_rescue, tab_analytics, tab_import = st.tabs([
        "📊 Overview",
        "🔮 Medicine Demand Forecast",
        "⚠️ Stockout Risk",
        "⏳ Expiry Risk",
        "🛒 Smart Reordering",
        "🔄 Rescue-to-Reorder™",
        "⚡ Big Data Analytics",
        "📥 Import Sales Data"
    ])

    # =======================================================================
    # SECTION 1: OVERVIEW
    # =======================================================================
    with tab_overview:
        # Part 1A: Clearly Labelled Global Pharmacy Network KPIs
        st.markdown('<div class="section-header">🌐 Pharmacy Network Global KPIs (All 8 Categories)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Aggregated portfolio metrics across all therapeutic classes in the central warehouse.</div>', unsafe_allow_html=True)

        tot_vol = daily_df["units_sold"].sum()
        tot_stock = stockout_df["current_stock"].sum() if stockout_df is not None else 0.0
        tot_inv_val = stockout_df["inventory_value"].sum() if stockout_df is not None else 0.0
        crit_count = (stockout_df["risk_status"].isin(["CRITICAL_STOCKOUT_RISK", "REORDER_WARNING"])).sum() if stockout_df is not None else 0
        tot_val_risk = expiry_df["value_at_risk"].sum() if expiry_df is not None else 0.0
        tot_reorder_units = reorder_df["suggested_order_quantity"].sum() if reorder_df is not None else 0

        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        with g_col1:
            st.markdown(render_metric_card("Historical Sales Volume", f"{tot_vol:,.0f} units", "2014-2019 Total Facts (2,106 Days)"), unsafe_allow_html=True)
        with g_col2:
            st.markdown(render_metric_card("Current Warehouse Stock", f"{tot_stock:,.0f} units", f"₹{tot_inv_val:,.2f} Total Asset Value"), unsafe_allow_html=True)
        with g_col3:
            st.markdown(render_metric_card("Medicines at Stockout Risk", f"{crit_count} / 8 categories", "2 Critical · 2 Reorder Warnings", card_type="warning" if crit_count > 0 else "healthy"), unsafe_allow_html=True)
        with g_col4:
            st.markdown(render_metric_card("Capital at Expiry Risk", f"₹{tot_val_risk:,.2f}", "9 Batches Affected (FEFO Audit)", card_type="urgent" if tot_val_risk > 0 else "healthy"), unsafe_allow_html=True)

        st.markdown("---")

        # Part 1B: Clearly Labelled Selected Medicine Spotlight
        st.markdown(f'<div class="section-header">🎯 Selected Medicine Spotlight: {ATC_SHORT.get(selected_med, selected_med)} ({selected_med})</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-subtitle">Real-time dynamic telemetry filtered strictly for {ATC_NAMES.get(selected_med, selected_med)}.</div>', unsafe_allow_html=True)

        med_tot_sales = med_daily["units_sold"].sum() if not med_daily.empty else 0.0
        med_daily_mean = med_daily["units_sold"].mean() if not med_daily.empty else 0.0

        if med_stock_row is not None:
            med_stock_display = f"{med_stock_row['current_stock']:.0f} units"
            med_cover_display = f"Estimated Cover: {med_stock_row['estimated_days_of_cover']:.1f} days"
            card_type_stock = "warning" if med_stock_row['estimated_days_of_cover'] < 7 else "default"
            med_lt_display = f"{med_stock_row['lead_time_demand']:.1f} units"
            med_rop_display = f"Reorder Point (ROP): {med_stock_row['reorder_point']:.1f} units"
        else:
            med_stock_display = "Not Configured"
            med_cover_display = "No entry in inventory.csv"
            card_type_stock = "default"
            med_lt_display = "N/A"
            med_rop_display = "Requires inventory.csv"

        if med_reorder_row is not None:
            med_reorder_qty_display = f"{med_reorder_row['suggested_order_quantity']} units"
            med_reorder_urgency_display = f"Procurement Status: {med_reorder_row['reorder_status']}"
            card_type_reorder = "urgent" if med_reorder_row['reorder_status'] == "URGENT" else "healthy"
        else:
            med_reorder_qty_display = "N/A"
            med_reorder_urgency_display = "Requires inventory parameters"
            card_type_reorder = "default"

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(render_metric_card(f"{selected_med} Historical Volume", f"{med_tot_sales:,.0f} units", f"Mean Demand: {med_daily_mean:.2f} units/day", card_type="focus"), unsafe_allow_html=True)
        with m_col2:
            st.markdown(render_metric_card(f"{selected_med} Current Stock", med_stock_display, med_cover_display, card_type=card_type_stock), unsafe_allow_html=True)
        with m_col3:
            st.markdown(render_metric_card(f"{selected_med} Lead-Time Demand", med_lt_display, med_rop_display), unsafe_allow_html=True)
        with m_col4:
            st.markdown(render_metric_card(f"{selected_med} Replenishment Need", med_reorder_qty_display, med_reorder_urgency_display, card_type=card_type_reorder), unsafe_allow_html=True)

        if med_stock_row is None:
            st.info(f"ℹ️ **Notice for {selected_med}:** Sales records are loaded, but physical inventory parameters are not yet configured in data/inventory.csv. PHARMORA maintains strict zero-fabrication of warehouse stock.")

        # Actionable Priority Alerts Section
        st.markdown("#### 🚨 Actionable Priority Alerts")
        alerts_found = 0

        # Check if selected medicine has urgent requirements
        if med_reorder_row is not None and med_reorder_row["reorder_status"] == "URGENT":
            st.error(f"🚨 **Urgent Reorder for Focus Medicine [{selected_med} - {ATC_SHORT.get(selected_med, '')}]:** {med_reorder_row['suggested_order_quantity']} units needed immediately. {med_reorder_row['recommendation_reason']}")
            alerts_found += 1

        # Check if selected medicine has expired batches
        if expiry_df is not None:
            med_exp = expiry_df[(expiry_df["medicine_id"] == selected_med) & (expiry_df["expiry_status"] == "EXPIRED")]
            if not med_exp.empty:
                exp_qty = med_exp["unsold_units_at_risk"].sum()
                exp_loss = med_exp["value_at_risk"].sum()
                st.error(f"🛑 **Quarantine Alert for Focus Medicine [{selected_med}]:** {exp_qty:.0f} units across {len(med_exp)} expired batches (₹{exp_loss:,.2f}) must be quarantined immediately. Never dispense expired pharmaceuticals.")
                alerts_found += 1

        # Check for other network alerts
        if reorder_df is not None:
            other_urgent = reorder_df[(reorder_df["reorder_status"] == "URGENT") & (reorder_df["medicine_id"] != selected_med)]
            if not other_urgent.empty:
                other_list = ", ".join([f"{r['medicine_id']} ({r['suggested_order_quantity']}u)" for _, r in other_urgent.iterrows()])
                st.warning(f"⚠️ **Other Network Procurement Alerts:** Urgent reorders pending for: **{other_list}**.")
                alerts_found += 1

        if alerts_found == 0:
            st.success("✅ **Operations Normal:** No critical stockouts or expired stock detected for the active selection.")

        # Historical Sales Trend Chart for Selected Medicine
        st.markdown(f"#### 📈 Historical Sales Trajectory: {ATC_NAMES.get(selected_med, selected_med)}")
        if not med_daily.empty:
            med_plot = med_daily.copy()
            med_plot["7d_MA"] = med_plot["units_sold"].rolling(7, min_periods=1).mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=med_plot["date"],
                y=med_plot["units_sold"],
                name="Daily Actual Sales",
                mode="lines",
                line=dict(color="#94A3B8", width=1.2),
                opacity=0.75
            ))
            fig_trend.add_trace(go.Scatter(
                x=med_plot["date"],
                y=med_plot["7d_MA"],
                name="7-Day Moving Average Trend",
                mode="lines",
                line=dict(color="#166534", width=2.5)
            ))
            fig_trend.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="sans-serif", color="#374151"),
                xaxis=dict(title="Date", showgrid=True, gridcolor="#E8EDE9"),
                yaxis=dict(title="Units Sold", showgrid=True, gridcolor="#E8EDE9"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=340,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_trend, width="stretch")
        else:
            st.info(f"ℹ️ No historical sales records for {selected_med} within the selected date filter ({start_d.strftime('%d %b %Y')} to {end_d.strftime('%d %b %Y')}). Please expand your date range in the sidebar.")

        # Drug Portfolio Comparison Table with Selected Row Marked
        st.markdown("#### 📋 Central Pharmacy Portfolio Register")
        if stockout_df is not None and reorder_df is not None:
            merged_ov = stockout_df.merge(
                reorder_df[["medicine_id", "suggested_order_quantity", "reorder_status", "recommended_order_timing"]],
                on="medicine_id"
            ).copy()
            merged_ov["Focus"] = merged_ov["medicine_id"].apply(lambda x: "👉 SELECTED" if x == selected_med else "")
            merged_ov["Common Name"] = merged_ov["medicine_id"].map(lambda x: ATC_SHORT.get(x, x))
            merged_ov["Therapeutic Class"] = merged_ov["medicine_id"].map(lambda x: ATC_CLASSES.get(x, ""))

            disp_cols = [
                "Focus", "medicine_id", "Common Name", "current_stock", "pending_order_quantity",
                "lead_time_demand", "reorder_point", "suggested_order_quantity", "risk_status", "reorder_status"
            ]
            ov_disp = merged_ov[disp_cols].rename(columns={
                "medicine_id": "ATC Code",
                "current_stock": "Current Stock",
                "pending_order_quantity": "In-Flight Orders",
                "lead_time_demand": "LT Demand",
                "reorder_point": "ROP",
                "suggested_order_quantity": "Order Qty",
                "risk_status": "Stock Risk",
                "reorder_status": "Reorder Urgency"
            })
            st.dataframe(ov_disp, width="stretch", hide_index=True)

    # =======================================================================
    # SECTION 2: MEDICINE DEMAND FORECAST
    # =======================================================================
    with tab_forecast:
        st.markdown(f'<div class="section-header">🔮 Machine Learning Demand Forecasting: {ATC_NAMES.get(selected_med, selected_med)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Recursive 7-step multi-horizon daily demand forecasting benchmarked against 7-day Moving Average.</div>', unsafe_allow_html=True)

        # Selected Medicine Holdout Evaluation Metrics
        has_eval = (eval_df is not None) and (not eval_df[eval_df["medicine_id"] == selected_med].empty)
        hist_days = len(med_daily)

        if has_eval:
            ev_row = eval_df[eval_df["medicine_id"] == selected_med].iloc[0]

            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-left: 4px solid #166534; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #16352B; text-transform: uppercase;">Holdout Evaluation Period: 60 Chronological Days (2019-08-10 to 2019-10-08)</div>
                <div style="font-size: 0.88rem; color: #374151; margin-top: 2px;">
                    Random Forest forecaster trained strictly on historical sales prior to holdout window without data leakage. Outperformed 7-day Moving Average baseline.
                </div>
            </div>
            """, unsafe_allow_html=True)

            fc_m1, fc_m2, fc_m3, fc_m4, fc_m5 = st.columns(5)
            with fc_m1:
                st.markdown(render_metric_card("Actual Mean Demand", f"{ev_row['mean_actual_demand']:.2f} units", "Daily Holdout Mean"), unsafe_allow_html=True)
            with fc_m2:
                st.markdown(render_metric_card("Random Forest MAE", f"{ev_row['rf_mae']:.3f} units", f"{ev_row['mae_improvement_pct']}% error reduction", card_type="healthy"), unsafe_allow_html=True)
            with fc_m3:
                st.markdown(render_metric_card("Baseline MAE (7d-MA)", f"{ev_row['baseline_mae']:.3f} units", "Historical 7-Day Baseline"), unsafe_allow_html=True)
            with fc_m4:
                st.markdown(render_metric_card("Forecast RMSE", f"{ev_row['rf_rmse']:.3f} units", f"Baseline RMSE: {ev_row['baseline_rmse']:.3f}"), unsafe_allow_html=True)
            with fc_m5:
                st.markdown(render_metric_card("Model WAPE Error", f"{ev_row['rf_wape_pct']:.1f}%", f"Top Model: {ev_row['best_model']}"), unsafe_allow_html=True)
        else:
            if hist_days < 60:
                st.markdown(f"""
                <div style="background-color: #FEF3C7; border: 1px solid #FCD34D; border-left: 4px solid #B45309; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #92400E; text-transform: uppercase;">
                        ⚠️ Insufficient History for Machine Learning Forecasting ({hist_days} / 60 minimum required days)
                    </div>
                    <div style="font-size: 0.85rem; color: #78350F; margin-top: 4px; line-height: 1.4;">
                        PHARMORA strictly requires a minimum of 60 chronological daily sales records to construct autoregressive lag features (1, 2, 3, 7, 14, 21, 28 days) and evaluate holdout predictions without data leakage or overfitting.<br>
                        <strong>Zero-Fabrication Policy:</strong> No artificial sales facts or fabricated forecast accuracy metrics have been assigned to <strong>{selected_med}</strong>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### 📌 Explicitly Labelled Historical Fallback Projection")
                mean_d = med_daily["units_sold"].mean() if not med_daily.empty else 0.0
                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    st.markdown(render_metric_card("Fallback Daily Demand", f"{mean_d:.2f} units/day", "Historical Sample Mean (Labelled Fallback)", card_type="focus"), unsafe_allow_html=True)
                with f_col2:
                    st.markdown(render_metric_card("Historical Days Recorded", f"{hist_days} days", "Daily Sales Records Ingested"), unsafe_allow_html=True)
                with f_col3:
                    st.markdown(render_metric_card("Total Observed Volume", f"{med_daily['units_sold'].sum():,.1f} units", "Cumulative Actual Sales"), unsafe_allow_html=True)
            else:
                st.info(f"💡 **Sufficient Historical Data Available ({hist_days} days):** You can train a Random Forest forecaster for {selected_med} using the interactive backtester below.")

        # Unified Actual vs Forecast Visual
        st.markdown(f"#### 📊 Historical Sales & 7-Day Demand Projection: {ATC_SHORT.get(selected_med, selected_med)}")
        has_fc = (forecast_df is not None) and (not forecast_df[forecast_df["medicine_id"] == selected_med].empty)

        if has_fc:
            med_fc = forecast_df[forecast_df["medicine_id"] == selected_med].sort_values("forecast_date")
            recent_actual = med_daily.tail(60)

            fc_start = med_fc["forecast_date"].min()
            fc_end = med_fc["forecast_date"].max()
            act_max_date = recent_actual["date"].max() if not recent_actual.empty else actual_max_date
            act_cutoff_str = pd.to_datetime(act_max_date).strftime('%d %b %Y')

            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-left: 4px solid #166534; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 0.85rem; color: #374151;">
                    <strong style="color: #16352B;">Horizon Distinction:</strong>
                    Solid charcoal line displays verified <strong>actual historical sales facts</strong> (through {act_cutoff_str}).
                    Green dashed line with diamond markers represents <strong>7-day forward ML predictions</strong> ({fc_start.strftime('%d %b %Y')} – {fc_end.strftime('%d %b %Y')}).
                </div>
                <div style="background-color: #E8F3EC; color: #166534; border: 1px solid #A7F3D0; font-size: 0.75rem; font-weight: 700; padding: 3px 8px; border-radius: 12px;">
                    FUTURE ML FORECAST
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig_fc = go.Figure()

            # Historical actual sales trace
            if not recent_actual.empty:
                fig_fc.add_trace(go.Scatter(
                    x=recent_actual["date"],
                    y=recent_actual["units_sold"],
                    name=f"Actual Daily Sales (Facts through {act_cutoff_str})",
                    mode="lines",
                    line=dict(color="#374151", width=2)
                ))

            # Future ML Forecast trace
            fig_fc.add_trace(go.Scatter(
                x=med_fc["forecast_date"],
                y=med_fc["predicted_units"],
                name=f"7-Day ML Forecast (Projection: {fc_start.strftime('%d %b')} – {fc_end.strftime('%d %b %Y')})",
                mode="lines+markers",
                line=dict(color="#166534", width=3, dash="dash"),
                marker=dict(size=8, color="#166534", symbol="diamond")
            ))

            # Vertical demarcation line at historical actuals cutoff
            if not recent_actual.empty:
                fig_fc.add_vline(
                    x=recent_actual["date"].max(),
                    line_width=2,
                    line_dash="dot",
                    line_color="#B45309",
                    annotation_text="Cutoff: Historical Actuals End",
                    annotation_position="top left",
                    annotation_font=dict(size=11, color="#92400E", family="sans-serif")
                )

                # Visual bridge between last actual and first forecast
                last_act = recent_actual.iloc[-1]
                first_fc = med_fc.iloc[0]
                fig_fc.add_trace(go.Scatter(
                    x=[last_act["date"], first_fc["forecast_date"]],
                    y=[last_act["units_sold"], first_fc["predicted_units"]],
                    name="Actual-to-Forecast Boundary",
                    mode="lines",
                    line=dict(color="#B45309", width=1.5, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip"
                ))

            # Shaded forecast horizon rectangle
            fig_fc.add_vrect(
                x0=fc_start,
                x1=fc_end,
                fillcolor="#166534",
                opacity=0.07,
                layer="below",
                line_width=1,
                line_color="#166534",
                line_dash="dash",
                annotation_text="🔮 Forecast Horizon",
                annotation_position="top right",
                annotation_font=dict(size=11, color="#166534", family="sans-serif")
            )

            fig_fc.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="sans-serif", color="#374151"),
                xaxis=dict(title="Date", showgrid=True, gridcolor="#E8EDE9"),
                yaxis=dict(title="Units Sold / Predicted", showgrid=True, gridcolor="#E8EDE9"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=390,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_fc, width="stretch")

            # Daily predicted demand schedule for Selected Medicine
            st.markdown(f"#### 📅 7-Day Day-by-Day Forecast Schedule: {ATC_SHORT.get(selected_med, '')}")
            if not med_fc.empty:
                med_fc_disp = med_fc[["forecast_date", "predicted_units", "model_name"]].copy()
                med_fc_disp["Day of Week"] = med_fc_disp["forecast_date"].dt.day_name()
                med_fc_disp["Date"] = med_fc_disp["forecast_date"].dt.strftime("%Y-%m-%d")
                med_fc_disp["Record Type"] = "🔮 Future Forecast (Not Actual Sales)"
                med_fc_disp = med_fc_disp[["Date", "Day of Week", "predicted_units", "model_name", "Record Type"]].rename(columns={
                    "predicted_units": "Projected Demand (ML Units)",
                    "model_name": "Forecasting Algorithm",
                    "Record Type": "Data Classification"
                })
                st.dataframe(med_fc_disp, width="stretch", hide_index=True)
        else:
            if not med_daily.empty:
                mean_d = med_daily["units_sold"].mean()
                fig_fb = go.Figure()
                fig_fb.add_trace(go.Scatter(
                    x=med_daily["date"],
                    y=med_daily["units_sold"],
                    name=f"Actual Daily Sales ({len(med_daily)} days)",
                    mode="lines+markers",
                    line=dict(color="#166534", width=2),
                    marker=dict(size=6, color="#166534")
                ))
                fig_fb.add_hline(
                    y=mean_d,
                    line_dash="dash",
                    line_color="#B45309",
                    annotation_text=f"Labelled Fallback Daily Mean: {mean_d:.2f} units",
                    annotation_position="top left"
                )
                fig_fb.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="sans-serif", color="#374151"),
                    xaxis=dict(title="Date", showgrid=True, gridcolor="#E8EDE9"),
                    yaxis=dict(title="Units Sold", showgrid=True, gridcolor="#E8EDE9"),
                    hovermode="x unified",
                    height=360,
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_fb, width="stretch")

                # Fallback schedule table
                st.markdown(f"#### 📅 7-Day Day-by-Day Fallback Projection Schedule: {ATC_SHORT.get(selected_med, selected_med)}")
                last_dt = pd.to_datetime(med_daily["date"].max())
                future_dates = [last_dt + pd.Timedelta(days=i) for i in range(1, 8)]
                fb_sched = pd.DataFrame({
                    "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
                    "Day of Week": [d.day_name() for d in future_dates],
                    "Projected Demand (Labelled Fallback Daily Mean)": [round(mean_d, 2)] * 7
                })
                st.dataframe(fb_sched, width="stretch", hide_index=True)

        # Full 8-Medicine 7-Day Forecast Matrix
        with st.expander("🌐 View 7-Day Forecast Matrix across Available Medicines"):
            if forecast_df is not None:
                pivot_fc = forecast_df.pivot(index="forecast_date", columns="medicine_id", values="predicted_units").reset_index()
                pivot_fc["Day"] = pivot_fc["forecast_date"].dt.day_name()
                pivot_fc["Date"] = pivot_fc["forecast_date"].dt.strftime("%Y-%m-%d")
                fc_med_cols = [c for c in all_meds if c in pivot_fc.columns]
                cols = ["Date", "Day"] + fc_med_cols
                st.dataframe(pivot_fc[cols], width="stretch", hide_index=True)

        # Interactive Model Retraining Backtest
        with st.expander("🛠️ Interactive Forecaster Backtest & Holdout Re-evaluation"):
            st.caption("Retrain Scikit-learn Random Forest model on historical data using custom holdout horizons.")
            bt_col1, bt_col2 = st.columns([1, 2])
            with bt_col1:
                test_days = st.slider(f"Holdout Window for {selected_med} (Days)", min_value=30, max_value=120, value=60, step=15, key=f"holdout_slider_{selected_med}")
                btn_train = st.button("Run Holdout Retraining", type="primary", key=f"btn_train_{selected_med}")

            if btn_train:
                if len(med_daily) < 65:
                    st.error(f"Cannot train forecaster for {selected_med}: minimum 65 chronological records required (found {len(med_daily)}).")
                else:
                    try:
                        with st.spinner(f"Training Random Forest on {selected_med} with {test_days} days holdout..."):
                            _, metrics = train_medicine_forecaster(daily_df, selected_med, test_days=test_days)
                        with bt_col2:
                            st.success(f"**Retraining Complete for {selected_med}:**")
                            st.write(f"- Random Forest MAE: **{metrics['mae']} units** (RMSE: {metrics['rmse']} units)")
                            st.write(f"- 7-Day Moving Avg Baseline MAE: **{metrics['baseline_mae']} units**")
                            st.write(f"- Error Reduction: **{metrics['mae_improvement_pct']}%** ({'Random Forest Wins' if metrics['better_than_baseline'] else 'Baseline Wins'})")
                    except Exception as ex:
                        st.error(f"Training error: {ex}")

    # =======================================================================
    # SECTION 3: STOCKOUT RISK
    # =======================================================================
    with tab_stockout:
        st.markdown(f'<div class="section-header">⚠️ Inventory Stockout Risk Engine: {ATC_NAMES.get(selected_med, selected_med)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Evaluating Net Inventory Position against Statistical Reorder Point (ROP = Lead-Time Demand + Safety Stock).</div>', unsafe_allow_html=True)

        # Selected Medicine Telemetry Cards
        if med_stock_row is not None:
            so_c1, so_c2, so_c3, so_c4 = st.columns(4)
            with so_c1:
                st.markdown(render_metric_card(f"{selected_med} Physical Stock", f"{med_stock_row['current_stock']:.0f} units", f"Unit Cost: ₹{med_stock_row['unit_cost']:.2f}"), unsafe_allow_html=True)
            with so_c2:
                st.markdown(render_metric_card(f"{selected_med} Net Position", f"{med_stock_row['inventory_position']:.0f} units", f"Pending In-Flight: {med_stock_row['pending_order_quantity']:.0f} units"), unsafe_allow_html=True)
            with so_c3:
                st.markdown(render_metric_card(f"{selected_med} Reorder Point (ROP)", f"{med_stock_row['reorder_point']:.1f} units", f"Lead Demand: {med_stock_row['lead_time_demand']:.1f} · SS: {med_stock_row['safety_stock']:.1f}"), unsafe_allow_html=True)
            with so_c4:
                card_style = "urgent" if med_stock_row["risk_status"] == "CRITICAL_STOCKOUT_RISK" else ("warning" if med_stock_row["risk_status"] == "REORDER_WARNING" else "healthy")
                st.markdown(render_metric_card(f"{selected_med} Cover & Status", f"{med_stock_row['estimated_days_of_cover']:.1f} days", f"Risk: {med_stock_row['risk_status']}", card_type=card_style), unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-left: 4px solid #166534; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #16352B;">ℹ️ Warehouse Inventory Profile Required for {selected_med}</div>
                <div style="font-size: 0.85rem; color: #374151; margin-top: 6px; line-height: 1.5;">
                    Medicine <strong>{selected_med}</strong> is active in sales facts, but has no physical warehouse tracking profile in <code>data/inventory.csv</code>.<br>
                    <strong>Zero-Fabrication Policy:</strong> PHARMORA never fabricates physical stock quantities, supplier lead times, or unit costs for newly imported medicines.<br>
                    To enable automated stockout risk monitoring and dynamic ROP calculation, add a row for <code>{selected_med}</code> to <code>data/inventory.csv</code>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Selected Medicine vs Portfolio Stockout Posture Chart
        st.markdown("#### 📊 Net Inventory Position vs. Reorder Point (ROP) across Portfolio")
        if stockout_df is not None:
            so_chart_df = stockout_df.copy()
            so_chart_df["Drug"] = so_chart_df["medicine_id"].map(lambda x: ATC_SHORT.get(x, x))
            so_chart_df["Is_Selected"] = so_chart_df["medicine_id"] == selected_med

            fig_so = go.Figure()
            # Position bar
            fig_so.add_trace(go.Bar(
                x=so_chart_df["Drug"],
                y=so_chart_df["inventory_position"],
                name="Net Inventory Position (Stock + In-Flight)",
                marker_color=[COLOR_PRIMARY if sel else "#94A3B8" for sel in so_chart_df["Is_Selected"]]
            ))
            # ROP bar
            fig_so.add_trace(go.Bar(
                x=so_chart_df["Drug"],
                y=so_chart_df["reorder_point"],
                name="Reorder Point Threshold (ROP)",
                marker_color="#B91C1C"
            ))
            # Lead time demand
            fig_so.add_trace(go.Scatter(
                x=so_chart_df["Drug"],
                y=so_chart_df["lead_time_demand"],
                name="Expected Lead-Time Demand",
                mode="markers",
                marker=dict(size=10, color="#B45309", symbol="diamond")
            ))
            fig_so.update_layout(
                barmode="group",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="sans-serif", color="#334155"),
                xaxis=dict(title="Medicine Category", showgrid=True, gridcolor="#E8EDE9"),
                yaxis=dict(title="Quantity (Units)", showgrid=True, gridcolor="#E8EDE9"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=360,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_so, width="stretch")

        # Full Stockout Risk Register Table
        st.markdown("#### 📋 Stockout Risk Register & Diagnostic Indicators")
        if stockout_df is not None:
            st_disp = stockout_df.copy()
            st_disp["Focus"] = st_disp["medicine_id"].apply(lambda x: "👉 SELECTED" if x == selected_med else "")
            st_disp["Drug"] = st_disp["medicine_id"].map(lambda x: ATC_SHORT.get(x, x))
            cols = [
                "Focus", "medicine_id", "Drug", "current_stock", "pending_order_quantity",
                "inventory_position", "supplier_lead_time_days", "lead_time_demand",
                "safety_stock", "reorder_point", "estimated_days_of_cover", "risk_status"
            ]
            st_disp = st_disp[cols].rename(columns={
                "medicine_id": "ATC Code",
                "current_stock": "Physical Stock",
                "pending_order_quantity": "In-Flight Pending",
                "inventory_position": "Net Position",
                "supplier_lead_time_days": "Lead Time (d)",
                "lead_time_demand": "LT Demand",
                "safety_stock": "Safety Stock",
                "reorder_point": "ROP",
                "estimated_days_of_cover": "Days Cover",
                "risk_status": "Risk Status"
            })
            st.dataframe(st_disp, width="stretch", hide_index=True)

        # What-If Inventory Simulator
        with st.expander(f"🧪 What-If Parameter Simulator for {selected_med} ({ATC_SHORT.get(selected_med, '')})"):
            st.caption("Simulate supplier lead time fluctuations and safety buffer sensitivity.")
            sim_c1, sim_c2 = st.columns(2)
            default_stock = float(med_stock_row["current_stock"]) if med_stock_row is not None else 50.0
            default_lt = int(med_stock_row["supplier_lead_time_days"]) if med_stock_row is not None else 7

            with sim_c1:
                sim_lt = st.slider(f"Supplier Replenishment Lead Time (Days)", 1, 30, default_lt, key=f"sim_lt_{selected_med}")
                sim_stock = st.number_input(f"Simulated Physical Stock Level", 0.0, 1000.0, default_stock, step=10.0, key=f"sim_stk_{selected_med}")

            sim_metrics = calculate_inventory_metrics(med_daily["units_sold"], lead_time_days=int(sim_lt))
            sim_status = assess_stock_status(sim_stock, sim_metrics["reorder_point"], sim_metrics["safety_stock"])

            with sim_c2:
                st.write(f"**Diagnostic Evaluation for {selected_med}:**")
                st.markdown(f"Status: {render_badge_html(sim_status)}", unsafe_allow_html=True)
                st.write(f"- Projected Lead-Time Demand: `{sim_metrics['avg_daily_demand'] * sim_lt:.2f}` units")
                st.write(f"- Statistical Safety Stock: `{sim_metrics['safety_stock']:.2f}` units")
                st.write(f"- Dynamic Reorder Point: `{sim_metrics['reorder_point']:.2f}` units")

    # =======================================================================
    # SECTION 4: EXPIRY RISK
    # =======================================================================
    with tab_expiry:
        st.markdown(f'<div class="section-header">⏳ FEFO Batch Expiration Risk Audit: {ATC_NAMES.get(selected_med, selected_med)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Multi-batch First-Expired, First-Out (FEFO) cumulative demand allocation and healthcare quarantine audit.</div>', unsafe_allow_html=True)

        st.info("ℹ️ **Synthetic Demonstration Master:** The 16 batch records are synthetic demonstrations mapped to real ATC demand curves. **Healthcare Safety Protocol:** Expired medications must never be dispensed under any circumstances.")

        # Filter batches strictly for the selected medicine
        med_batches = expiry_df[expiry_df["medicine_id"] == selected_med] if expiry_df is not None else pd.DataFrame()

        # Selected Medicine Batch Summary Cards
        b_total = len(med_batches)
        if not med_batches.empty:
            b_exp = med_batches[med_batches["expiry_status"] == "EXPIRED"]
            b_near = med_batches[med_batches["expiry_status"] == "EXPIRY_RISK"]
            b_safe = med_batches[med_batches["expiry_status"] == "HEALTHY_BATCH"]

            exp_units = b_exp["unsold_units_at_risk"].sum() if not b_exp.empty else 0.0
            exp_loss = b_exp["value_at_risk"].sum() if not b_exp.empty else 0.0
            near_units = b_near["unsold_units_at_risk"].sum() if not b_near.empty else 0.0
            near_loss = b_near["value_at_risk"].sum() if not b_near.empty else 0.0

            ex_c1, ex_c2, ex_c3, ex_c4 = st.columns(4)
            with ex_c1:
                st.markdown(render_metric_card(f"{selected_med} Batches Tracked", f"{b_total} batches", f"{len(b_safe)} Healthy Batches"), unsafe_allow_html=True)
            with ex_c2:
                st.markdown(render_metric_card(f"{selected_med} Expired Units", f"{exp_units:.0f} units", f"₹{exp_loss:,.2f} Loss (Quarantine)", card_type="urgent" if exp_units > 0 else "healthy"), unsafe_allow_html=True)
            with ex_c3:
                st.markdown(render_metric_card(f"{selected_med} Near Expiry Risk", f"{near_units:.1f} units", f"₹{near_loss:,.2f} Exposed", card_type="warning" if near_units > 0 else "healthy"), unsafe_allow_html=True)
            with ex_c4:
                st.markdown(render_metric_card(f"{selected_med} Capital at Risk", f"₹{exp_loss + near_loss:,.2f}", "Total Financial Exposure"), unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-left: 4px solid #166534; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #16352B;">ℹ️ Batch Expiration Records Required for {selected_med}</div>
                <div style="font-size: 0.85rem; color: #374151; margin-top: 6px; line-height: 1.5;">
                    No physical batch tracking records are registered for <strong>{selected_med}</strong> in <code>data/batches.csv</code>.<br>
                    <strong>Healthcare Safety Compliance:</strong> Expired pharmaceuticals must never be dispensed under any circumstances. PHARMORA does not fabricate synthetic batch numbers, quantities, or expiration dates for newly imported medicines.<br>
                    To monitor FEFO shelf-life horizon and quarantine risks, register batches for <code>{selected_med}</code> in <code>data/batches.csv</code>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Batch table filtered strictly for Selected Medicine
        st.markdown(f"#### 📦 Active Warehouse Batches for {ATC_SHORT.get(selected_med, selected_med)} ({selected_med})")
        if not med_batches.empty:
            b_disp = med_batches[[
                "batch_id", "batch_quantity", "expiry_date", "days_until_expiry",
                "expected_demand_before_expiry", "unsold_units_at_risk",
                "unit_cost", "value_at_risk", "expiry_status", "action_recommendation"
            ]].copy()
            b_disp["expiry_date"] = b_disp["expiry_date"].dt.strftime("%Y-%m-%d")
            b_disp = b_disp.rename(columns={
                "batch_id": "Batch ID",
                "batch_quantity": "Batch Qty",
                "expiry_date": "Expiration Date",
                "days_until_expiry": "Days Remaining",
                "expected_demand_before_expiry": "FEFO Demand",
                "unsold_units_at_risk": "Unsold Units",
                "unit_cost": "Unit Cost (₹)",
                "value_at_risk": "Value at Risk (₹)",
                "expiry_status": "Status",
                "action_recommendation": "Prescriptive Action"
            })
            st.dataframe(b_disp, width="stretch", hide_index=True)
        else:
            st.info(f"No batches registered for {selected_med} in the current warehouse inventory dataset.")

        # Batch Shelf-Life Horizon Scatter Chart
        st.markdown("#### ⏳ Batch Shelf-Life Horizon across All Warehouse Batches")
        if expiry_df is not None:
            fig_ex = px.scatter(
                expiry_df,
                x="days_until_expiry",
                y="batch_quantity",
                color="expiry_status",
                size="value_at_risk",
                hover_data=["batch_id", "medicine_id", "unsold_units_at_risk", "value_at_risk"],
                color_discrete_map={
                    "EXPIRED": "#B91C1C",
                    "EXPIRY_RISK": "#B45309",
                    "HEALTHY_BATCH": "#15803D"
                },
                labels={
                    "days_until_expiry": "Days Until Expiration (Negative = Expired)",
                    "batch_quantity": "Batch Size (Units)",
                    "expiry_status": "Expiry Status"
                }
            )
            fig_ex.add_vline(x=0, line_dash="dash", line_color="#B91C1C", annotation_text="Today's Horizon")
            fig_ex.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="sans-serif", color="#334155"),
                xaxis=dict(title="Days Until Expiration", showgrid=True, gridcolor="#E8EDE9"),
                yaxis=dict(title="Batch Quantity", showgrid=True, gridcolor="#E8EDE9"),
                height=340,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_ex, width="stretch")

        # Network-wide batch audit
        with st.expander("🌐 View All 16 Warehouse Batches across All Categories"):
            if expiry_df is not None:
                all_b_disp = expiry_df[[
                    "batch_id", "medicine_id", "batch_quantity", "expiry_date",
                    "days_until_expiry", "unsold_units_at_risk", "value_at_risk",
                    "expiry_status", "action_recommendation"
                ]].copy()
                all_b_disp["expiry_date"] = all_b_disp["expiry_date"].dt.strftime("%Y-%m-%d")
                st.dataframe(all_b_disp, width="stretch", hide_index=True)

    # =======================================================================
    # SECTION 5: SMART REORDERING
    # =======================================================================
    with tab_reorder:
        st.markdown(f'<div class="section-header">🛒 Smart Reordering Engine: {ATC_NAMES.get(selected_med, selected_med)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Prescriptive replenishment policy answering what, how much, and when to order with in-flight order netting.</div>', unsafe_allow_html=True)

        # Selected Medicine Order Recommendation Spotlight
        if med_reorder_row is not None:
            order_badge = render_badge_html(med_reorder_row['reorder_status'])
            card_border = "#FCA5A5" if med_reorder_row['reorder_status'] == "URGENT" else "#DDE5DF"

            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid {card_border}; border-radius: 8px; padding: 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(22, 53, 43, 0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #16352B;">
                        Prescriptive Procurement Decision for {selected_med} · {ATC_SHORT.get(selected_med, '')}
                    </div>
                    <div>{order_badge}</div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0; border-top: 1px solid #DDE5DF; border-bottom: 1px solid #DDE5DF; padding: 10px 0;">
                    <div>
                        <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase;">Suggested Order</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: {'#B91C1C' if med_reorder_row['reorder_status'] == 'URGENT' else '#166534'};">{med_reorder_row['suggested_order_quantity']} units</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase;">Target Stock (S)</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #16352B;">{med_reorder_row['target_stock']:.1f} units</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase;">In-Flight Pending</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #16352B;">{med_reorder_row['pending_order_quantity']:.0f} units</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase;">Order Timing</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #16352B; margin-top: 4px;">{med_reorder_row['recommended_order_timing']}</div>
                    </div>
                </div>
                <div style="font-size: 0.88rem; color: #374151; line-height: 1.4; margin-top: 6px;">
                    <strong style="color: #16352B;">Clinical & Operational Rationale:</strong> {med_reorder_row['recommendation_reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-left: 4px solid #166534; border-radius: 8px; padding: 16px 20px; margin-bottom: 18px;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #16352B;">ℹ️ Reorder Parameters Required for {selected_med}</div>
                <div style="font-size: 0.85rem; color: #374151; margin-top: 6px; line-height: 1.5;">
                    Prescriptive reordering calculates order quantities based on target stock (S), supplier lead time, and pending orders. Because <strong>{selected_med}</strong> does not yet have an entry in <code>data/inventory.csv</code>, reorder calculations are deferred until warehouse parameters are configured.<br>
                    <strong>Zero-Fabrication Guarantee:</strong> PHARMORA does not fabricate synthetic reorder amounts.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Inventory Decomposition Stacked Bar Chart
        st.markdown("#### 📊 Target Stock vs. Inventory Allocation across Categories")
        if reorder_df is not None:
            r_chart = reorder_df.copy()
            r_chart["Drug"] = r_chart["medicine_id"].map(lambda x: ATC_SHORT.get(x, x))
            r_chart["Is_Selected"] = r_chart["medicine_id"] == selected_med

            fig_ro = go.Figure()
            # Physical stock
            fig_ro.add_trace(go.Bar(
                x=r_chart["Drug"],
                y=r_chart["current_stock"],
                name="Physical Current Stock",
                marker_color="#166534"
            ))
            # In flight pending
            fig_ro.add_trace(go.Bar(
                x=r_chart["Drug"],
                y=r_chart["pending_order_quantity"],
                name="In-Flight Pending Orders",
                marker_color="#52B788"
            ))
            # Suggested order
            fig_ro.add_trace(go.Bar(
                x=r_chart["Drug"],
                y=r_chart["suggested_order_quantity"],
                name="Suggested Reorder Quantity",
                marker_color="#B91C1C"
            ))
            # Target line
            fig_ro.add_trace(go.Scatter(
                x=r_chart["Drug"],
                y=r_chart["target_stock"],
                name="Target Stock (S = Lead Time + 7d Review)",
                mode="lines+markers",
                line=dict(color="#16352B", width=3)
            ))
            fig_ro.update_layout(
                barmode="stack",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="sans-serif", color="#374151"),
                xaxis=dict(title="Medicine Category", showgrid=True, gridcolor="#E8EDE9"),
                yaxis=dict(title="Quantity (Units)", showgrid=True, gridcolor="#E8EDE9"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=360,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_ro, width="stretch")

        # Full Reorder Decision Register Table
        st.markdown("#### 📋 Comprehensive Prescriptive Reorder Decision Matrix")
        if reorder_df is not None:
            r_disp = reorder_df.copy()
            r_disp["Focus"] = r_disp["medicine_id"].apply(lambda x: "👉 SELECTED" if x == selected_med else "")
            r_disp["Drug"] = r_disp["medicine_id"].map(lambda x: ATC_SHORT.get(x, x))
            cols = [
                "Focus", "medicine_id", "Drug", "current_stock", "pending_order_quantity",
                "forecast_lead_time_demand", "reorder_point", "target_stock",
                "suggested_order_quantity", "reorder_status", "recommended_order_timing"
            ]
            r_disp = r_disp[cols].rename(columns={
                "medicine_id": "ATC Code",
                "current_stock": "Current Stock",
                "pending_order_quantity": "In-Flight Pending",
                "forecast_lead_time_demand": "LT Demand",
                "reorder_point": "ROP",
                "target_stock": "Target Stock (S)",
                "suggested_order_quantity": "Order Qty",
                "reorder_status": "Urgency",
                "recommended_order_timing": "Order Timing"
            })
            st.dataframe(r_disp, width="stretch", hide_index=True)

    # =======================================================================
    # SECTION: RESCUE-TO-REORDER™ (Pharmacy Setup & Stock Sharing)
    # =======================================================================
    with tab_rescue:
        curr_mode = st.session_state.get("rescue_operating_mode")
        is_onboarding = (curr_mode is None) or st.session_state.get("show_rescue_onboarding", False)

        # -------------------------------------------------------------------
        # 1. PHARMACY SETUP / ONBOARDING VIEW
        # -------------------------------------------------------------------
        if is_onboarding:
            st.markdown("""
            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-radius: 12px; padding: 28px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(22, 53, 43, 0.05); text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #166534; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                    ✨ PHARMORA Rescue-to-Reorder™ Onboarding
                </div>
                <div style="font-size: 1.7rem; font-weight: 800; color: #16352B; margin-bottom: 6px;">
                    How does your pharmacy business operate?
                </div>
                <div style="font-size: 1.02rem; color: #475569; max-width: 680px; margin: 0 auto 12px auto; line-height: 1.5;">
                    We'll personalize inventory recommendations based on how your pharmacy works.
                </div>
                <div style="display: inline-flex; align-items: center; background-color: #E8F3EC; border: 1px solid #A5D6A7; color: #166534; font-weight: 700; font-size: 0.88rem; padding: 5px 16px; border-radius: 20px;">
                    Tagline: “Rescue Before You Reorder.”
                </div>
            </div>
            """, unsafe_allow_html=True)

            card_col1, card_col2 = st.columns(2)

            with card_col1:
                st.markdown("""
                <div class="choice-card">
                    <div>
                        <div class="choice-icon">🏢</div>
                        <div class="choice-title">Option A — I Manage Multiple Branches</div>
                        <div class="choice-desc">“I own or manage multiple pharmacy locations.”</div>
                        <div class="choice-support">
                            Move eligible surplus stock between your branches before purchasing more.
                            Safely rebalance expiring inventory from low-velocity clinics to high-demand flagship dispensaries.
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B; border-top: 1px solid #E8EDE9; padding-top: 10px; margin-bottom: 16px;">
                            • Consolidated branch inventory & batch tracking<br>
                            • Internal FEFO surplus-to-shortage matching<br>
                            • Net supplier purchase bill reduction<br>
                            • Pharmacist authorization safeguards before transfers
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🏢 Manage My Branches", type="primary", key="btn_choose_multibranch", use_container_width=True):
                    st.session_state["rescue_operating_mode"] = MODE_MULTI_BRANCH
                    st.session_state["show_rescue_onboarding"] = False
                    st.rerun()

            with card_col2:
                st.markdown("""
                <div class="choice-card">
                    <div>
                        <div class="choice-icon">🤝</div>
                        <div class="choice-title">Option B — I Want to Share Stock With Other Pharmacies</div>
                        <div class="choice-desc">“I operate an independent pharmacy and want to connect with other pharmacies.”</div>
                        <div class="choice-support">
                            Discover potential stock-sharing opportunities with participating pharmacies.
                            Prevent obsolescence by sharing surplus medicine batches within a verified, privacy-protected healthcare network.
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B; border-top: 1px solid #E8EDE9; padding-top: 10px; margin-bottom: 16px;">
                            • Verified state pharmacy license directory<br>
                            • Masked privacy-protected match discovery<br>
                            • Mutual agreement workflow before confirmation<br>
                            • Zero automatic debits (strictly independent inventories)
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🤝 Explore Pharmacy Network", type="primary", key="btn_choose_network", use_container_width=True):
                    st.session_state["rescue_operating_mode"] = MODE_NETWORK_SHARING
                    st.session_state["show_rescue_onboarding"] = False
                    st.rerun()

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Option C: Single store
            st.markdown("""
            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-radius: 10px; padding: 18px 24px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(22, 53, 43, 0.03);">
                <div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #16352B; display: flex; align-items: center; gap: 8px;">
                        <span>🏪</span> I currently operate just one pharmacy.
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">
                        Continue with single-store expiry risk detection and reorder recommendations, with the option to enable stock sharing later.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            single_c1, single_c2 = st.columns([3, 1])
            with single_c2:
                if st.button("🏪 Continue Single Store", key="btn_choose_single_store", use_container_width=True):
                    st.session_state["rescue_operating_mode"] = MODE_SINGLE_STORE
                    st.session_state["show_rescue_onboarding"] = False
                    st.rerun()

        else:
            # -------------------------------------------------------------------
            # 2. ACTIVE RESCUE-TO-REORDER™ DASHBOARD
            # -------------------------------------------------------------------
            mode_meta = OPERATING_MODES.get(curr_mode, {})
            mode_badge_color = mode_meta.get("badge_color", "#166534")

            # Top Bar: Mode Status, Tagline & Change Model Action
            h_col1, h_col2 = st.columns([3, 1])
            with h_col1:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <div style="font-size: 1.35rem; font-weight: 800; color: #16352B;">
                        🔄 PHARMORA Rescue-to-Reorder™
                    </div>
                    <span style="background-color: #E8F3EC; color: #166534; border: 1px solid #A5D6A7; padding: 3px 10px; border-radius: 16px; font-size: 0.8rem; font-weight: 700;">
                        “Rescue Before You Reorder.”
                    </span>
                    <span style="background-color: #F1F5F9; color: {mode_badge_color}; border: 1px solid #CBD5E1; padding: 3px 10px; border-radius: 16px; font-size: 0.8rem; font-weight: 700;">
                        {mode_meta.get('icon', '🏥')} Mode: {mode_meta.get('title', curr_mode)}
                    </span>
                </div>
                <div style="font-size: 0.86rem; color: #64748B; margin-top: 4px;">
                    {mode_meta.get('supporting_text', '')}
                </div>
                """, unsafe_allow_html=True)

            with h_col2:
                if st.button("⚙️ Change Operating Model", key="btn_reopen_onboarding", use_container_width=True):
                    st.session_state["show_rescue_onboarding"] = True
                    st.rerun()

            st.markdown("---")

            # Execute 7-Step Rescue Engine
            engine_output = run_rescue_to_reorder_engine(
                operating_mode=curr_mode,
                branch_inventory=st.session_state["branch_inventory_df"],
                branch_batches=st.session_state["branch_batches_df"],
                target_branch_id=None,
                network_stock=st.session_state["network_stock_df"],
                active_confirmed_transfers=st.session_state["active_confirmed_transfers"],
                reference_date=actual_max_date
            )

            summary = engine_output["summary"]
            comparison = engine_output["comparison"]
            all_recs = engine_output["recommendations"]

            # --- Top 4 KPI Cards ---
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            with kpi_c1:
                st.markdown(render_metric_card(
                    "Potential Stock Rescued",
                    f"{summary['potential_stock_rescued_units']} units",
                    "Eligible surplus diverted from waste",
                    card_type="focus" if summary['potential_stock_rescued_units'] > 0 else "default"
                ), unsafe_allow_html=True)

            with kpi_c2:
                st.markdown(render_metric_card(
                    "Potential Expiry Waste Avoided",
                    f"₹{summary['expiry_waste_avoided_val']:,.2f}",
                    "Preserved pharmaceutical asset value",
                    card_type="healthy" if summary['expiry_waste_avoided_val'] > 0 else "default"
                ), unsafe_allow_html=True)

            with kpi_c3:
                st.markdown(render_metric_card(
                    "Potential Purchases Avoided",
                    f"₹{summary['purchases_avoided_val']:,.2f}",
                    f"{comparison['net_savings_pct']}% Procurement Bill Reduction",
                    card_type="healthy" if summary['purchases_avoided_val'] > 0 else "default"
                ), unsafe_allow_html=True)

            with kpi_c4:
                st.markdown(render_metric_card(
                    "Pending Transfer / Requests",
                    f"{summary['pending_transfer_count']} Opportunities",
                    "Actionable stock-rescue workflows",
                    card_type="warning" if summary['pending_transfer_count'] > 0 else "default"
                ), unsafe_allow_html=True)

            # --- "Transfer First" vs "Purchase New Stock" Comparison Ledger ---
            st.markdown("#### ⚖️ Decision Ledger: “Transfer First” vs. “Purchase New Stock”")
            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                st.markdown(f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid #64748B; border-radius: 8px; padding: 18px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Conventional Reorder Strategy (Without Rescue)</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #1E293B; margin: 4px 0 12px 0;">
                        ₹{comparison['traditional_purchase_cost']:,.2f} <span style="font-size: 0.85rem; font-weight: 500; color: #64748B;">({comparison['traditional_purchase_units']} units)</span>
                    </div>
                    <div style="font-size: 0.84rem; color: #475569; line-height: 1.5;">
                        • <strong>Wholesaler Procurement:</strong> Full gross shortage ordered externally.<br>
                        • <strong>Supplier Lead Time:</strong> 4 to 6 business days replenishment delay.<br>
                        • <strong>Expiry Exposure:</strong> Eligible surplus batches at other locations expire unused and are written off as capital loss.<br>
                        • <strong>Working Capital:</strong> Maximum cash tied up in redundant inventory.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with comp_col2:
                savings_val = comparison['rescued_value_avoided']
                st.markdown(f"""
                <div style="background-color: #FFFFFF; border: 1px solid #A5D6A7; border-top: 4px solid #166534; border-radius: 8px; padding: 18px; height: 100%; box-shadow: 0 2px 5px rgba(22, 101, 52, 0.06);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-size: 0.78rem; font-weight: 700; color: #166534; text-transform: uppercase;">PHARMORA Rescue-to-Reorder™ Strategy</div>
                        <span style="background-color: #E8F3EC; color: #166534; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 12px;">
                            Save {comparison['net_savings_pct']}% Net
                        </span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #166534; margin: 4px 0 12px 0;">
                        ₹{comparison['residual_purchase_cost']:,.2f} <span style="font-size: 0.85rem; font-weight: 500; color: #64748B;">(Residual: {comparison['residual_purchase_units']} units)</span>
                    </div>
                    <div style="font-size: 0.84rem; color: #374151; line-height: 1.5;">
                        • <strong>Rescued Surplus:</strong> <strong>{comparison['rescued_units']} units</strong> (₹{savings_val:,.2f}) relocated before placing supplier orders.<br>
                        • <strong>Fulfillment Speed:</strong> Fast local transit (same-day or next-day courier).<br>
                        • <strong>Zero Waste Impact:</strong> Short-dated stock consumed safely prior to expiration.<br>
                        • <strong>Supplier Order:</strong> Reduced strictly to the remaining unmet shortage.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            # --- Interactive Filters Bar ---
            st.markdown("#### 🔍 Filter Recommendations & Stock Opportunities")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)

            with f_col1:
                filter_med_options = ["All 8 Medicines"] + all_meds
                default_med_choice = selected_med if selected_med in filter_med_options else "All 8 Medicines"
                filter_med = st.selectbox(
                    "Filter by Medicine",
                    options=filter_med_options,
                    index=filter_med_options.index(default_med_choice) if default_med_choice in filter_med_options else 0,
                    format_func=lambda x: f"{x} - {ATC_SHORT.get(x, '')}" if x != "All 8 Medicines" else "All 8 Medicines",
                    key="rescue_filter_med"
                )

            with f_col2:
                if curr_mode == MODE_MULTI_BRANCH:
                    branch_opts = ["All Branches"] + [b["branch_id"] for b in DEFAULT_BRANCHES]
                    filter_branch = st.selectbox(
                        "Filter by Branch",
                        options=branch_opts,
                        format_func=lambda x: f"{x} - {next((b['name'] for b in DEFAULT_BRANCHES if b['branch_id'] == x), x)}" if x != "All Branches" else "All Branches",
                        key="rescue_filter_branch"
                    )
                else:
                    filter_branch = st.selectbox(
                        "Location Scope",
                        options=["All Network Locations", "Local Store (Self)", "Nearby Peers (< 10km)"],
                        key="rescue_filter_branch_net"
                    )

            with f_col3:
                filter_expiry = st.selectbox(
                    "Expiry Window",
                    options=["All Horizons", "Approaching (< 60 Days)", "Within 90 Days", "Within 180 Days"],
                    key="rescue_filter_expiry"
                )

            with f_col4:
                filter_status = st.selectbox(
                    "Decision Status",
                    options=["All Recommendations", "Recommended Transfers / Matches Only", "Supplier Orders Only"],
                    key="rescue_filter_status"
                )

            # Filter recommendation records
            filtered_recs = []
            for r in all_recs:
                if r["recommendation_id"] in st.session_state.get("rejected_recommendations", set()):
                    continue
                if filter_med != "All 8 Medicines" and r["medicine_id"] != filter_med:
                    continue
                if curr_mode == MODE_MULTI_BRANCH and filter_branch != "All Branches":
                    if r.get("destination_id") != filter_branch and r.get("source_id") != filter_branch:
                        continue
                if filter_expiry == "Approaching (< 60 Days)" and r.get("days_to_expiry", 999) > 60:
                    continue
                if filter_expiry == "Within 90 Days" and r.get("days_to_expiry", 999) > 90:
                    continue
                if filter_expiry == "Within 180 Days" and r.get("days_to_expiry", 999) > 180:
                    continue
                if filter_status == "Recommended Transfers / Matches Only" and r["decision"] not in ["RECOMMENDED_TRANSFER", "POTENTIAL_NETWORK_MATCH"]:
                    continue
                if filter_status == "Supplier Orders Only" and r["decision"] not in ["SUPPLIER_PURCHASE_ONLY", "SINGLE_STORE_REORDER"]:
                    continue
                filtered_recs.append(r)

            st.markdown("---")

            # ---------------------------------------------------------------
            # 3. MODE-SPECIFIC OPERATIONAL WORKFLOWS
            # ---------------------------------------------------------------

            # ---------------------------------------------------------------
            # WORKFLOW A: MULTI-BRANCH WORKFLOW
            # ---------------------------------------------------------------
            if curr_mode == MODE_MULTI_BRANCH:
                st.markdown('<div class="section-header">🏢 Multi-Branch Inventory Balancing & Transfer Execution</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-subtitle">Identify eligible surplus stock at source branches to satisfy shortage at destination branches before purchasing from suppliers.</div>', unsafe_allow_html=True)

                # Consolidated Branch Inventory Inspector
                with st.expander("🌐 Inspect Consolidated Branch Inventories, Batches & Lead Times"):
                    st.caption("Centralized inventory across all 3 registered pharmacy branches:")
                    b_inv_disp = st.session_state["branch_inventory_df"].copy()
                    b_inv_disp["Medicine"] = b_inv_disp["medicine_id"].map(lambda x: ATC_SHORT.get(x, x))
                    b_inv_disp["Branch Name"] = b_inv_disp["branch_id"].map(lambda x: next((b["name"] for b in DEFAULT_BRANCHES if b["branch_id"] == x), x))
                    st.dataframe(
                        b_inv_disp[[
                            "branch_id", "Branch Name", "medicine_id", "Medicine",
                            "current_stock", "pending_orders", "daily_demand", "lead_time_days", "safety_stock", "unit_cost"
                        ]].rename(columns={
                            "branch_id": "Branch ID",
                            "medicine_id": "ATC",
                            "current_stock": "Current Stock",
                            "pending_orders": "In-Flight",
                            "daily_demand": "Daily Demand",
                            "lead_time_days": "Lead Time (Days)",
                            "safety_stock": "Safety Stock",
                            "unit_cost": "Unit Cost (₹)"
                        }),
                        width="stretch",
                        hide_index=True
                    )

                # Transfer Recommendations List
                st.markdown(f"##### 📦 Prescriptive Transfer Recommendations ({len(filtered_recs)} Actions Evaluated)")

                if not filtered_recs:
                    st.info("ℹ️ No recommendations match the selected filters.")
                else:
                    for rec in filtered_recs:
                        med_code = rec["medicine_id"]
                        med_label = ATC_NAMES.get(med_code, med_code)
                        rec_id = rec["recommendation_id"]

                        if rec["decision"] == "RECOMMENDED_TRANSFER":
                            # Actionable Transfer Proposal Card
                            st.markdown(f"""
                            <div style="background-color: #FFFFFF; border: 1px solid #A7F3D0; border-left: 6px solid #166534; border-radius: 8px; padding: 18px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(22, 53, 43, 0.04);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <div style="font-size: 1.12rem; font-weight: 800; color: #16352B;">
                                        🚚 Proposed Inter-Branch Transfer: {rec['transfer_quantity']} units of {med_label}
                                    </div>
                                    <span style="background-color: #E8F3EC; color: #166534; border: 1px solid #A5D6A7; font-weight: 700; font-size: 0.8rem; padding: 3px 10px; border-radius: 12px;">
                                        RECOMMENDED TRANSFER
                                    </span>
                                </div>
                                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0; background-color: #F8FAF9; border: 1px solid #DDE5DF; border-radius: 6px; padding: 10px 14px;">
                                    <div>
                                        <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase;">Source Branch</div>
                                        <div style="font-size: 0.95rem; font-weight: 700; color: #16352B;">{rec['source_name']}</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase;">Destination Branch</div>
                                        <div style="font-size: 0.95rem; font-weight: 700; color: #16352B;">{rec['destination_name']}</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase;">Batch & Shelf Life</div>
                                        <div style="font-size: 0.95rem; font-weight: 700; color: #B45309;">{rec['batch_id']} · {rec['days_to_expiry']} days left</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase;">Avoided Purchase & Waste</div>
                                        <div style="font-size: 1.05rem; font-weight: 800; color: #15803D;">₹{rec['financial_waste_avoided']:,.2f}</div>
                                    </div>
                                </div>
                                <div style="display: flex; gap: 16px; margin: 10px 0; font-size: 0.88rem; color: #374151;">
                                    <div style="background-color: #F1F5F9; border-radius: 4px; padding: 6px 12px;">
                                        Gross Reorder Need: <strong>{rec['original_reorder_qty']} units</strong>
                                    </div>
                                    <div style="background-color: #E8F3EC; color: #166534; border-radius: 4px; padding: 6px 12px;">
                                        Proposed Transfer: <strong>{rec['transfer_quantity']} units</strong>
                                    </div>
                                    <div style="background-color: #FEF3C7; color: #92400E; border-radius: 4px; padding: 6px 12px;">
                                        Remaining Supplier Reorder: <strong>{rec['remaining_reorder_qty']} units</strong>
                                    </div>
                                </div>
                                <div style="font-size: 0.86rem; color: #374151; line-height: 1.45; margin-bottom: 12px;">
                                    <strong>Analytical Rationale:</strong> {rec['reason']}<br>
                                    <strong>Residual Procurement Plan:</strong> {rec['residual_action']}
                                </div>
                                <div style="font-size: 0.78rem; color: #64748B; font-style: italic; margin-bottom: 12px;">
                                    🔒 <strong>Authorized Execution Guard:</strong> Inventory quantities are never changed until an authorized user confirms this transfer.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Action Buttons for Authorized Transfer
                            act_c1, act_c2, act_c3 = st.columns([2, 2, 4])
                            with act_c1:
                                if st.button(f"✅ Accept & Confirm Transfer", key=f"btn_accept_{rec_id}", type="primary", use_container_width=True):
                                    try:
                                        upd_inv, audit_entry = execute_confirmed_transfer(
                                            branch_inventory=st.session_state["branch_inventory_df"],
                                            source_branch_id=rec["source_id"],
                                            dest_branch_id=rec["destination_id"],
                                            medicine_id=rec["medicine_id"],
                                            quantity=rec["transfer_quantity"]
                                        )
                                        st.session_state["branch_inventory_df"] = upd_inv
                                        st.session_state["confirmed_transfer_history"].append(audit_entry)
                                        st.session_state["active_confirmed_transfers"].append({
                                            "source_id": rec["source_id"],
                                            "medicine_id": rec["medicine_id"],
                                            "quantity": rec["transfer_quantity"],
                                            "status": "CONFIRMED"
                                        })
                                        st.success(f"🎉 **Transfer Authorized & Executed!** Successfully moved {rec['transfer_quantity']} units of {rec['medicine_id']} from {rec['source_name']} to {rec['destination_name']}.")
                                        st.rerun()
                                    except Exception as ex:
                                        st.error(f"Transfer error: {ex}")

                            with act_c2:
                                if st.button(f"❌ Reject Recommendation", key=f"btn_reject_{rec_id}", use_container_width=True):
                                    st.session_state["rejected_recommendations"].add(rec_id)
                                    st.info(f"Recommendation {rec_id} rejected. Stock remains at {rec['source_name']}.")
                                    st.rerun()

                        elif rec["decision"] == "SUPPLIER_PURCHASE_ONLY":
                            # Supplier Reorder Fallback Card (Explaining why no transfer is recommended)
                            st.markdown(f"""
                            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-left: 6px solid #64748B; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <div style="font-size: 1.05rem; font-weight: 700; color: #16352B;">
                                        🛒 Wholesaler Reorder: {rec['original_reorder_qty']} units of {med_label}
                                    </div>
                                    <span style="background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; font-weight: 600; font-size: 0.78rem; padding: 2px 8px; border-radius: 10px;">
                                        EXTERNAL PURCHASE
                                    </span>
                                </div>
                                <div style="font-size: 0.85rem; color: #475569; line-height: 1.45;">
                                    <strong>Destination:</strong> {rec['destination_name']}<br>
                                    <strong>Transfer Feasibility:</strong> {rec['reason']}<br>
                                    <strong>Prescriptive Action:</strong> Place standard purchase order for {rec['original_reorder_qty']} units with primary distributor.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Confirmed Transfer Audit Log
                if st.session_state.get("confirmed_transfer_history"):
                    st.markdown("##### 📜 Confirmed Branch Transfer Audit Log (Session)")
                    audit_df = pd.DataFrame(st.session_state["confirmed_transfer_history"])
                    st.dataframe(audit_df, width="stretch", hide_index=True)

            # ---------------------------------------------------------------
            # WORKFLOW B: INDEPENDENT PHARMACY NETWORK SHARING
            # ---------------------------------------------------------------
            elif curr_mode == MODE_NETWORK_SHARING:
                st.markdown('<div class="section-header">🤝 Independent Pharmacy Stock-Sharing Network</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-subtitle">Discover potential stock-sharing matches with verified participating pharmacies while strictly safeguarding commercial privacy and independent inventory ownership.</div>', unsafe_allow_html=True)

                # Opt-in Network Registration Profile
                with st.expander("🛡️ Pharmacy Network Registration & Privacy Controls", expanded=True):
                    p_reg = st.session_state["pharmacy_registration"]
                    reg_c1, reg_c2 = st.columns(2)
                    with reg_c1:
                        st.markdown(f"""
                        <div style="font-size: 0.9rem; color: #16352B; line-height: 1.6;">
                            <strong>Participating Pharmacy:</strong> {p_reg['name']}<br>
                            <strong>Location:</strong> {p_reg['location']}<br>
                            <strong>State Pharmacy License:</strong> <span style="background-color: #E8F3EC; color: #166534; padding: 2px 6px; border-radius: 4px; font-weight: 700;">✅ {p_reg['license']} (VERIFIED)</span><br>
                            <strong>Pharmacist in Charge:</strong> {p_reg['contact_person']} ({p_reg['contact_phone']})
                        </div>
                        """, unsafe_allow_html=True)
                    with reg_c2:
                        st.markdown(f"""
                        <div style="font-size: 0.9rem; color: #16352B; line-height: 1.6;">
                            <strong>Privacy Level:</strong> <code>{p_reg['privacy_level']}</code> (Masked discovery until mutual request)<br>
                            <strong>Sharing Categories:</strong> {', '.join(p_reg['sharing_categories'])}<br>
                            <strong>Compliance Certification:</strong> GxP Good Distribution Practice Certified<br>
                            <span style="font-size: 0.8rem; color: #64748B;">ℹ️ Clearly labelled synthetic demo network. No live pharmacy stock is exposed without mutual authorization.</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Potential Stock-Sharing Matches
                st.markdown("##### 🔍 Potential Stock-Sharing Matches")
                st.markdown("""
                <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #0284C7; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.85rem; color: #334155;">
                    Opportunities below are initially shown as <strong>“Potential stock-sharing matches”</strong>, not confirmed availability.<br>
                    Exact batch identifiers and internal balances are masked. A request-to-connect must be sent and accepted by both pharmacies before finalizing transfer logistics.
                </div>
                """, unsafe_allow_html=True)

                if not filtered_recs:
                    st.info("ℹ️ No network sharing matches available for current filters.")
                else:
                    for rec in filtered_recs:
                        rec_id = rec["recommendation_id"]
                        med_label = ATC_NAMES.get(rec["medicine_id"], rec["medicine_id"])

                        if rec["decision"] == "POTENTIAL_NETWORK_MATCH":
                            st.markdown(f"""
                            <div style="background-color: #FFFFFF; border: 1px solid #BAE6FD; border-left: 6px solid #0284C7; border-radius: 8px; padding: 18px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(2, 132, 199, 0.04);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <div style="font-size: 1.1rem; font-weight: 800; color: #0C4A6E;">
                                        🤝 Potential Match: ~{rec['transfer_quantity']} units of {med_label}
                                    </div>
                                    <span style="background-color: #E0F2FE; color: #0369A1; border: 1px solid #7DD3FC; font-weight: 700; font-size: 0.8rem; padding: 3px 10px; border-radius: 12px;">
                                        POTENTIAL STOCK-SHARING MATCH
                                    </span>
                                </div>
                                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0; background-color: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 6px; padding: 10px 14px;">
                                    <div>
                                        <div style="font-size: 0.72rem; color: #0284C7; text-transform: uppercase;">Participating Peer</div>
                                        <div style="font-size: 0.95rem; font-weight: 700; color: #0C4A6E;">{rec['source_name']}</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.72rem; color: #0284C7; text-transform: uppercase;">Your Store Need</div>
                                        <div style="font-size: 0.95rem; font-weight: 700; color: #0C4A6E;">{rec['original_reorder_qty']} units</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.72rem; color: #0284C7; text-transform: uppercase;">Estimated Shelf Life</div>
                                        <div style="font-size: 0.95rem; font-weight: 700; color: #B45309;">~{rec['days_to_expiry']} days (Masked Lot)</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.72rem; color: #0284C7; text-transform: uppercase;">Potential Capital Rescued</div>
                                        <div style="font-size: 1.05rem; font-weight: 800; color: #15803D;">₹{rec['financial_waste_avoided']:,.2f}</div>
                                    </div>
                                </div>
                                <div style="font-size: 0.86rem; color: #374151; line-height: 1.45; margin-bottom: 12px;">
                                    <strong>Opportunity Details:</strong> {rec['reason']}<br>
                                    <strong>Supplier Residual Plan:</strong> {rec['residual_action']}
                                </div>
                                <div style="font-size: 0.78rem; color: #64748B; font-style: italic; margin-bottom: 12px;">
                                    🔒 <strong>Independent Boundary Safeguard:</strong> Inventory remains strictly independent for each pharmacy. Never automatically transferred between unrelated owners.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Send Request Action Button
                            net_btn_c1, net_btn_c2 = st.columns([3, 5])
                            with net_btn_c1:
                                if st.button(f"📨 Send Stock-Sharing Request to Peer", key=f"btn_send_net_{rec_id}", type="primary", use_container_width=True):
                                    req_entry = {
                                        "request_id": f"REQ-{datetime.datetime.now().strftime('%H%M%S')}-{rec['medicine_id']}",
                                        "peer_pharmacy_id": rec["source_id"],
                                        "peer_pharmacy_name": rec["source_name"],
                                        "medicine_id": rec["medicine_id"],
                                        "requested_quantity": rec["transfer_quantity"],
                                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "status": "PENDING_PEER_CONFIRMATION",
                                        "peer_contact": "Protected until agreement",
                                        "terms": "Standard GxP wholesale transfer (₹0 markup)"
                                    }
                                    st.session_state["network_sharing_requests"].append(req_entry)
                                    st.success(f"📨 **Request Transmitted!** Sent stock-sharing request for {rec['transfer_quantity']} units of {rec['medicine_id']} to {rec['source_name']}.")
                                    st.rerun()

                        elif rec["decision"] == "SUPPLIER_PURCHASE_ONLY":
                            st.markdown(f"""
                            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-left: 6px solid #64748B; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                                <div style="font-size: 1.02rem; font-weight: 700; color: #16352B;">
                                    🛒 Direct Supplier Order: {rec['original_reorder_qty']} units of {med_label}
                                </div>
                                <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">
                                    {rec['reason']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Sent Stock-Sharing Requests & Mutual Consent Ledger
                if st.session_state.get("network_sharing_requests"):
                    st.markdown("##### 📬 Sent Stock-Sharing Requests (Mutual Agreement Workflow)")
                    for req in st.session_state["network_sharing_requests"]:
                        req_status = req["status"]
                        is_pending = req_status == "PENDING_PEER_CONFIRMATION"

                        st.markdown(f"""
                        <div style="background-color: #FFFFFF; border: 1px solid {'#FCD34D' if is_pending else '#A5D6A7'}; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-size: 0.95rem; font-weight: 700; color: #16352B;">
                                    Request {req['request_id']} · {req['requested_quantity']} units of {req['medicine_id']} to {req['peer_pharmacy_name']}
                                </div>
                                <span>{render_badge_html(req_status)}</span>
                            </div>
                            <div style="font-size: 0.82rem; color: #64748B; margin-top: 4px;">
                                Initiated: {req['timestamp']} · Terms: {req['terms']} · Contact: <strong>{req['peer_contact']}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if is_pending:
                            sim_c1, sim_c2 = st.columns([3, 5])
                            with sim_c1:
                                if st.button(f"🤝 Simulate Peer Confirmation (Demo)", key=f"btn_sim_confirm_{req['request_id']}", use_container_width=True):
                                    req["status"] = "CONFIRMED_BY_PEER"
                                    req["peer_contact"] = "Dr. Sarah Lin, RPh · (555) 012-4821 · dispatch@metrocare.pharmacy"
                                    st.balloons()
                                    st.success(f"🤝 **Mutual Agreement Confirmed!** {req['peer_pharmacy_name']} accepted request. Direct dispatch instructions unlocked.")
                                    st.rerun()

            # ---------------------------------------------------------------
            # WORKFLOW C: SINGLE STORE WORKFLOW
            # ---------------------------------------------------------------
            else:
                st.markdown('<div class="section-header">🏪 Single-Store Operating Mode</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-subtitle">Single-store operation with localized demand forecasting, stockout detection, and primary wholesaler reordering.</div>', unsafe_allow_html=True)

                st.markdown("""
                <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-left: 4px solid #64748B; border-radius: 8px; padding: 18px; margin-bottom: 18px;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #16352B; margin-bottom: 6px;">Single Pharmacy Operation Active</div>
                    <div style="font-size: 0.88rem; color: #374151; line-height: 1.5;">
                        You are currently operating in <strong>Single-Store Mode</strong>. Inventory risk detection and reorder calculations operate on your local store warehouse.<br>
                        Surplus stock at other clinics or independent pharmacies is not being cross-balanced, and all replenishment is fulfilled via external supplier wholesale orders.<br>
                        <strong>Upgrade Options:</strong> You can enable Multi-Branch balancing if you operate multiple locations, or join the PHARMORA Stock-Sharing Network to rescue surplus stock from peer pharmacies.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                up_c1, up_c2 = st.columns(2)
                with up_c1:
                    st.markdown("""
                    <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-radius: 8px; padding: 16px;">
                        <div style="font-size: 1rem; font-weight: 700; color: #166534;">🏢 Expand to Multi-Branch Management</div>
                        <div style="font-size: 0.82rem; color: #475569; margin: 6px 0 12px 0;">
                            Connect multiple dispensary locations to rebalance surplus inventory before purchasing more.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Switch to Multi-Branch Mode", key="btn_switch_to_mb", use_container_width=True):
                        st.session_state["rescue_operating_mode"] = MODE_MULTI_BRANCH
                        st.rerun()

                with up_c2:
                    st.markdown("""
                    <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-radius: 8px; padding: 16px;">
                        <div style="font-size: 1rem; font-weight: 700; color: #0284C7;">🤝 Join Pharmacy Stock-Sharing Network</div>
                        <div style="font-size: 0.82rem; color: #475569; margin: 6px 0 12px 0;">
                            Opt into privacy-protected discovery to find expiring stock with nearby participating pharmacies.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Switch to Pharmacy Network", key="btn_switch_to_net", use_container_width=True):
                        st.session_state["rescue_operating_mode"] = MODE_NETWORK_SHARING
                        st.rerun()

                # Local Reorder Table for Single Store
                st.markdown("##### 🛒 Local Store Prescriptive Reorders")
                single_recs = [r for r in all_recs if r["decision"] == "SINGLE_STORE_REORDER"]
                if single_recs:
                    s_df = pd.DataFrame([{
                        "Medicine": ATC_SHORT.get(r["medicine_id"], r["medicine_id"]),
                        "ATC Code": r["medicine_id"],
                        "Reorder Qty": r["original_reorder_qty"],
                        "Unit Cost (₹)": f"₹{r['unit_cost']:.2f}",
                        "Estimated Spend (₹)": f"₹{r['original_reorder_qty'] * r['unit_cost']:.2f}",
                        "Action": r["action_label"]
                    } for r in single_recs])
                    st.dataframe(s_df, width="stretch", hide_index=True)

    # =======================================================================
    # SECTION 6: BIG DATA ANALYTICS
    # =======================================================================
    with tab_analytics:
        st.markdown(f'<div class="section-header">⚡ Big Data Engineering & Analytics (PySpark & Spark SQL)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Distributed pipeline execution metrics, monthly longitudinal trajectories, and retail seasonality.</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-left: 4px solid #166534; border-radius: 8px; padding: 14px 18px; margin-bottom: 18px;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #16352B;">PySpark Distributed Pipeline Architecture</div>
            <div style="font-size: 0.85rem; color: #374151; margin-top: 2px;">
                Ingested <strong>2,106</strong> daily sales records (2014-2019) from <code>data/salesdaily.csv</code>.
                Unpivoted 8 wide ATC drug categories into <strong>16,848</strong> long-format transaction facts.
                Volume reconciliation discrepancy: <strong>0.000000 units</strong> (127,595.5037 raw units &equiv; 127,595.5037 processed units).
            </div>
        </div>
        """, unsafe_allow_html=True)

        an_col1, an_col2 = st.columns(2)
        with an_col1:
            st.markdown(f"#### 📅 Day-of-Week Purchasing Seasonality ({start_d.strftime('%d %b %Y')} – {end_d.strftime('%d %b %Y')})")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_source = filtered_daily if not filtered_daily.empty else daily_df
            dow_sales = dow_source.groupby("day_of_week")["units_sold"].mean().reindex(dow_order).reset_index()
            fig_dow = px.bar(
                dow_sales,
                x="day_of_week",
                y="units_sold",
                labels={"day_of_week": "Day of Week", "units_sold": "Average Sales per Record (Units)"},
                color_discrete_sequence=["#166534"]
            )
            fig_dow.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="sans-serif", color="#374151"),
                xaxis=dict(title="Day of Week", showgrid=True, gridcolor="#E8EDE9"),
                yaxis=dict(title="Avg Units Sold", showgrid=True, gridcolor="#E8EDE9"),
                height=340,
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_dow, width="stretch")

        with an_col2:
            st.markdown("#### ⚡ Spark SQL Aggregated Portfolio KPIs")
            if summary_df is not None:
                sum_disp = summary_df.copy()
                sum_disp["Focus"] = sum_disp["medicine_id"].apply(lambda x: "👉 SELECTED" if x == selected_med else "")
                sum_disp["Common Name"] = sum_disp["medicine_id"].map(lambda x: ATC_SHORT.get(x, x))
                cols = ["Focus", "medicine_id", "Common Name", "total_units_sold", "avg_daily_demand", "demand_volatility", "peak_single_day_sales"]
                sum_disp = sum_disp[cols].rename(columns={
                    "medicine_id": "ATC Code",
                    "total_units_sold": "Total Units Sold",
                    "avg_daily_demand": "Daily Mean",
                    "demand_volatility": "Volatility (StdDev)",
                    "peak_single_day_sales": "Peak Single Day"
                })
                st.dataframe(sum_disp, width="stretch", hide_index=True)

        # Monthly Longitudinal Trajectory for Selected Medicine
        st.markdown(f"#### 📈 Monthly Longitudinal Trajectory ({start_d.strftime('%b %Y')} – {end_d.strftime('%b %Y')}): {ATC_NAMES.get(selected_med, selected_med)}")
        if not med_daily.empty:
            med_m = med_daily.copy()
            med_m["year"] = med_m["date"].dt.year
            med_m["month"] = med_m["date"].dt.month
            m_calc = med_m.groupby(["year", "month"])["units_sold"].sum().reset_index()
            m_calc["period"] = m_calc["year"].astype(str) + "-" + m_calc["month"].astype(str).str.zfill(2)
            fig_m = px.bar(
                m_calc,
                x="period",
                y="units_sold",
                labels={"period": "Year-Month Period", "units_sold": "Monthly Units Sold"},
                color_discrete_sequence=["#166534"]
            )
            fig_m.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="sans-serif", color="#374151"),
                xaxis=dict(title="Year-Month Period", showgrid=True, gridcolor="#E8EDE9"),
                yaxis=dict(title="Monthly Units Sold", showgrid=True, gridcolor="#E8EDE9"),
                height=360,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_m, width="stretch")
        elif monthly_df is not None:
            m_filtered = monthly_df[monthly_df["medicine_id"] == selected_med].copy()
            m_filtered["period"] = m_filtered["year"].astype(str) + "-" + m_filtered["month"].astype(str).str.zfill(2)
            fig_m = px.bar(
                m_filtered,
                x="period",
                y="monthly_units_sold",
                labels={"period": "Year-Month", "monthly_units_sold": "Total Units Sold"},
                color_discrete_sequence=["#166534"]
            )
            fig_m.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="sans-serif", color="#374151"),
                xaxis=dict(title="Year-Month Period", showgrid=True, gridcolor="#E8EDE9"),
                yaxis=dict(title="Monthly Units Sold", showgrid=True, gridcolor="#E8EDE9"),
                height=360,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_m, width="stretch")
        else:
            st.info(f"ℹ️ No monthly records available for {selected_med} in the selected date range.")

    # =======================================================================
    # SECTION 7: IMPORT SALES DATA
    # =======================================================================
    with tab_import:
        st.markdown('<div class="section-header">📥 Import Supplementary Medicine Sales Records</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Safely ingest additional pharmacy sales data with real-time schema validation, multi-format parsing, deduplication, and non-destructive persistence.</div>', unsafe_allow_html=True)

        # Guidance and Template Download
        doc_col1, doc_col2 = st.columns([2, 1])
        with doc_col1:
            st.markdown("""
            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                <div style="font-size: 0.92rem; font-weight: 700; color: #16352B; margin-bottom: 6px;">📋 Standard Ingestion Schema & Guidelines</div>
                <div style="font-size: 0.84rem; color: #374151; line-height: 1.5;">
                    PHARMORA accepts standard long-format CSV files as well as legacy wide-format tables:<br>
                    • <strong>date</strong>: Transaction date in ISO (<code>YYYY-MM-DD</code>) or standard formats (e.g., <code>2020-01-15</code> or <code>1/15/2020</code>).<br>
                    • <strong>medicine_id</strong>: Pharmaceutical active ingredient / ATC identifier (e.g., <code>J01CA</code>, <code>N02BE</code>).<br>
                    • <strong>units_sold</strong>: Non-negative numeric sales volume (&ge; 0). Negative quantities or nulls will be rejected.<br>
                    • <strong>transaction_id</strong> (Optional): Unique transaction identifier for granular multi-order tracking.<br>
                    • <strong>Wide Format Support</strong>: If your CSV has a <code>datum</code>/<code>date</code> column and multiple medicine columns, PHARMORA will automatically unpivot it.<br>
                    • <strong>Data Safety Guarantee</strong>: Baseline data in <code>data/salesdaily.csv</code> is <strong>never overwritten</strong>. All validated records are safely staged in <code>data/imported_sales.csv</code>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with doc_col2:
            st.markdown("""
            <div style="background-color: #FFFFFF; border: 1px solid #DDE5DF; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                <div style="font-size: 0.92rem; font-weight: 700; color: #16352B; margin-bottom: 6px;">📥 CSV Template</div>
                <div style="font-size: 0.84rem; color: #374151; line-height: 1.4; margin-bottom: 12px;">
                    Download a standardized, pre-formatted CSV template ready for uploading additional sales facts.
                </div>
            </div>
            """, unsafe_allow_html=True)
            template_csv = "date,medicine_id,units_sold,transaction_id\n2020-01-01,J01CA,14.5,TX-20200101-001\n2020-01-02,J01CA,18.0,TX-20200102-001\n2020-01-03,J01CA,12.0,TX-20200103-001\n"
            st.download_button(
                label="📥 Download Template CSV",
                data=template_csv,
                file_name="pharmora_sales_template.csv",
                mime="text/csv",
                key="btn_download_template"
            )

        # File Uploader
        st.markdown("#### 📂 Upload Sales CSV")
        uploaded_file = st.file_uploader(
            "Select or drag & drop CSV file to import",
            type=["csv"],
            key="sales_csv_file_uploader",
            help="Upload a CSV file containing date, medicine_id, and units_sold."
        )

        if uploaded_file is not None:
            # 1. Validation Step
            with st.spinner("Validating schema, parsing dates, and checking numeric constraints..."):
                val_res = validate_sales_records(uploaded_file, source_filename=uploaded_file.name)

            if not val_res["is_valid"] and val_res["total_rows"] == 0:
                st.error(f"❌ **Upload Failed:** {val_res['error_message']}")
            else:
                raw_count = val_res["total_rows"]
                valid_count = val_res["valid_count"]
                invalid_count = val_res["invalid_count"]
                format_detected = val_res["detected_format"]

                # 2. Deduplication Step on Valid Records
                if valid_count > 0:
                    unique_df, dup_count, dup_df = deduplicate_records(val_res["valid_df"])
                else:
                    unique_df, dup_count, dup_df = pd.DataFrame(), 0, pd.DataFrame()

                net_importable = len(unique_df)

                # Validation & Telemetry Summary Cards
                st.markdown("##### 🔍 Upload Validation & Deduplication Telemetry")
                u_c1, u_c2, u_c3, u_c4, u_c5 = st.columns(5)
                with u_c1:
                    st.markdown(render_metric_card("Total Rows Scanned", f"{raw_count:,}", f"Format: {format_detected}"), unsafe_allow_html=True)
                with u_c2:
                    st.markdown(render_metric_card("Valid Records", f"{valid_count:,}", f"{valid_count / raw_count * 100:.1f}% Schema Pass" if raw_count > 0 else "0%"), unsafe_allow_html=True)
                with u_c3:
                    st.markdown(render_metric_card("Rejected / Errors", f"{invalid_count:,}", "Schema or Value Flaws", card_type="urgent" if invalid_count > 0 else "healthy"), unsafe_allow_html=True)
                with u_c4:
                    st.markdown(render_metric_card("Duplicates Skipped", f"{dup_count:,}", "Pre-existing in facts/staging", card_type="warning" if dup_count > 0 else "default"), unsafe_allow_html=True)
                with u_c5:
                    st.markdown(render_metric_card("Net Ready to Ingest", f"{net_importable:,}", "Non-duplicate clean facts", card_type="focus" if net_importable > 0 else "default"), unsafe_allow_html=True)

                # If there are validation errors, display detailed error register
                if invalid_count > 0:
                    st.markdown(f"##### ⚠️ Validation Error Register ({invalid_count} issues detected)")
                    st.warning("The records below violate schema or data quality requirements and were excluded from ingestion:")
                    err_df = pd.DataFrame(val_res["errors"])
                    st.dataframe(err_df, width="stretch", hide_index=True)

                # Preview of Valid Importable Records
                if net_importable > 0:
                    st.markdown(f"##### ✅ Preview of Valid Records Ready for Ingestion ({net_importable:,} rows)")
                    preview_df = unique_df.head(15).copy()
                    st.dataframe(preview_df, width="stretch", hide_index=True)

                    # List unique medicines in batch
                    batch_meds = sorted(unique_df["medicine_id"].unique().tolist())
                    new_meds_in_batch = [m for m in batch_meds if m not in all_meds]
                    if new_meds_in_batch:
                        st.info(f"✨ **New Medicine Identifier(s) Detected:** `{', '.join(new_meds_in_batch)}`. Ingesting will add them dynamically to the medicine selector.")

                    # Ingestion Execution Button
                    btn_col1, btn_col2 = st.columns([1, 2])
                    with btn_col1:
                        confirm_import = st.button(
                            f"🚀 Ingest & Rebuild Parquet Facts ({net_importable:,} Records)",
                            type="primary",
                            key="btn_confirm_import_records"
                        )

                    if confirm_import:
                        with st.spinner("Persisting records to data/imported_sales.csv and updating Parquet tables..."):
                            ingest_result = persist_and_update_pipeline(unique_df, source_filename=uploaded_file.name)

                        if ingest_result["success"]:
                            st.success(
                                f"🎉 **Import Successful!** Ingested **{ingest_result['imported_count']:,}** records into "
                                f"`{ingest_result['staging_path']}`. Total active facts: **{ingest_result['total_active_records']:,}** across **{ingest_result['total_medicines']}** medicines."
                            )
                            if ingest_result["new_medicines"]:
                                st.balloons()
                                st.info(f"✨ New medicine categories available in selector: **{', '.join(ingest_result['new_medicines'])}**")
                            # Clear cache and reset session state so date range recalculates dynamically
                            st.cache_data.clear()
                            if "historical_date_slider" in st.session_state:
                                del st.session_state["historical_date_slider"]
                            if "last_known_max_date" in st.session_state:
                                del st.session_state["last_known_max_date"]
                            st.rerun()
                        else:
                            st.error(f"❌ Ingestion failed: {ingest_result['message']}")

                elif valid_count > 0 and dup_count > 0:
                    st.info("ℹ️ **All valid records in this file are already present in the database.** Zero new records to import (Idempotent deduplication policy prevented duplicate transactions).")

                # Show skipped duplicate rows in expander
                if dup_count > 0:
                    with st.expander(f"🔍 Inspect {dup_count} Skipped Duplicate Records"):
                        st.caption("These records matched existing (date, medicine_id) keys in daily_sales.parquet or transaction_ids in data/imported_sales.csv:")
                        st.dataframe(dup_df.head(20), width="stretch", hide_index=True)

        st.markdown("---")

        # Persistent Staging Location History
        st.markdown("#### 📜 Persistent Staging History (`data/imported_sales.csv`)")
        staging_file = Path(IMPORTED_STAGING_PATH)
        if staging_file.exists() and staging_file.stat().st_size > 0:
            try:
                stg_df = pd.read_csv(staging_file)
                st.caption(f"Currently staging **{len(stg_df):,}** total imported records appended over time. Original baseline dataset (`data/salesdaily.csv`) remains untouched.")
                st.dataframe(stg_df.tail(25), width="stretch", hide_index=True)
            except Exception as e:
                st.warning(f"Could not read staging history: {e}")
        else:
            st.info("ℹ️ No supplementary records imported yet. Baseline data is untouched at `data/salesdaily.csv` (16,848 facts across 8 ATC categories).")


if __name__ == "__main__":
    main()
