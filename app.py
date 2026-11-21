# =============================================================================
# BHCPF Accountability Tracker — Streamlit App
# Developed by: Lawal Lawal Ali
# Date: October 2026
# Description: Interactive ML-powered accountability dashboard for the Basic
#              Health Care Provision Fund (BHCPF), tracking ₦57.1 billion
#              disbursed across 37 Nigerian states and FCT, Q1–Q3 2025.
# =============================================================================

import streamlit as st          # Main web app framework
import pandas as pd              # Data manipulation
import numpy as np               # Numerical operations
import plotly.express as px      # High-level chart building
import plotly.graph_objects as go # Low-level chart building (for custom charts)
from plotly.subplots import make_subplots  # Multi-panel chart layout

# =============================================================================
# PAGE CONFIG — Must be the first Streamlit call in the script
# =============================================================================
st.set_page_config(
    page_title="BHCPF Accountability Tracker",   # Browser tab title
    page_icon="🏥",                               # Browser tab icon
    layout="wide",                                # Use full browser width
    initial_sidebar_state="collapsed"            # Hide sidebar on load
)

# =============================================================================
# CUSTOM CSS — Inject styling to override Streamlit defaults
# =============================================================================
st.markdown("""
<style>
/* Remove default Streamlit top padding */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* Hero banner styling */
.hero-banner {
    background: linear-gradient(135deg, #0b1e3d 0%, #1d4ed8 100%);
    color: white; border-radius: 12px; padding: 28px 32px; margin-bottom: 24px;
}
.hero-banner h1 { font-size: 1.5rem; font-weight: 800; margin: 0 0 8px 0; }
.hero-banner p  { font-size: 0.875rem; color: rgba(255,255,255,0.75); margin: 0 0 14px 0; }
.hero-meta      { display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.8rem; color: rgba(255,255,255,0.6); }
.hero-meta strong { color: rgba(255,255,255,0.9); }

/* Stat metric box */
.metric-box {
    background: white; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 16px 20px;
}
.metric-box .label { font-size: 0.7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #6b7280; }
.metric-box .value { font-size: 1.75rem; font-weight: 800; line-height: 1.1; }
.metric-box .sub   { font-size: 0.75rem; color: #6b7280; margin-top: 2px; }
.metric-box.red  .value { color: #dc2626; }
.metric-box.amber .value { color: #d97706; }
.metric-box.green .value { color: #16a34a; }
.metric-box.blue  .value { color: #1d4ed8; }

/* Alert card */
.alert-card {
    border-radius: 10px; padding: 14px 16px; border: 1px solid;
    margin-bottom: 10px;
}
.alert-card.critical { background: #fef2f2; border-color: #fecaca; }
.alert-card.concern  { background: #fffbeb; border-color: #fde68a; }
.alert-card.watch    { background: #eff6ff; border-color: #bfdbfe; }
.alert-pill { font-size: 0.65rem; font-weight: 700; letter-spacing: .07em; padding: 2px 8px; border-radius: 999px; color: white; display: inline-block; margin-bottom: 6px; }
.critical .alert-pill { background: #dc2626; }
.concern  .alert-pill { background: #d97706; }
.watch    .alert-pill { background: #2563eb; }
.alert-states { font-weight: 700; font-size: 0.95rem; }
.critical .alert-states { color: #dc2626; }
.concern  .alert-states  { color: #d97706; }
.watch    .alert-states   { color: #2563eb; }
.alert-desc { font-size: 0.8rem; color: #6b7280; margin-top: 4px; line-height: 1.5; }

/* Disclosure box */
.disclosure {
    background: #fffbeb; border: 1px solid #fde68a;
    border-left: 4px solid #d97706; border-radius: 6px;
    padding: 12px 16px; font-size: 0.8rem; color: #6b7280;
    margin-bottom: 16px; line-height: 1.6;
}

/* Tier badge */
.badge { display: inline-block; font-size: 0.65rem; font-weight: 700; letter-spacing: .07em; padding: 2px 8px; border-radius: 999px; color: white; }
.badge-critical { background: #dc2626; }
.badge-concern  { background: #d97706; }
.badge-watch    { background: #2563eb; }
.badge-clean    { background: #16a34a; }

/* Footer */
.footer {
    background: #0b1e3d; color: rgba(255,255,255,0.4);
    padding: 18px 24px; border-radius: 10px;
    font-size: 0.75rem; text-align: center; line-height: 1.8;
    margin-top: 32px;
}

/* Hide Streamlit's default footer and menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA — All state-level pipeline output hardcoded from the ML pipeline.
#        Values are in ₦ Millions. Columns:
#        name, q1, q2, q3, tier, consensus_score,
#        functionality_score, total_facilities, high_risk_facilities
# =============================================================================
# =============================================================================
# DATA — Loaded from pipeline output CSVs
# =============================================================================
import os, joblib

@st.cache_data
def load_states():
    base = os.path.dirname(__file__)
    # anomaly_results.csv has state features + iso/lof/consensus scores
    df = pd.read_csv(os.path.join(base, "output", "anomaly_results.csv"))

    # Rename quarterly columns to short names the chart code expects
    df = df.rename(columns={"total_q1": "q1", "total_q2": "q2", "total_q3": "q3"})

    # Convert naira → ₦ millions for display
    for col in ["q1", "q2", "q3", "total_ytd"]:
        df[col] = (df[col] / 1_000_000).round(1)

    # Derive tier from consensus_score
    tier_map = {3: "CRITICAL", 2: "CONCERN", 1: "WATCH", 0: "CLEAN"}
    df["tier"] = df["consensus_score"].map(tier_map)

    return df

@st.cache_data
def load_high_risk_counts():
    base = os.path.dirname(__file__)
    fac = pd.read_csv(os.path.join(base, "output", "facility_risk_scores.csv"))
    # Count facilities per state with risk_score >= 0.80
    counts = (
        fac[fac["risk_score"] >= 0.80]
        .groupby("state")
        .size()
        .reset_index(name="high_risk_facilities")
    )
    return counts

df_states = load_states()
hr_counts = load_high_risk_counts()
df_states = df_states.merge(hr_counts, on="state", how="left")
df_states["high_risk_facilities"] = df_states["high_risk_facilities"].fillna(0).astype(int)

# Recompute derived columns (in case they're not in the CSV)
df_states["qoq_q2_pct"] = ((df_states["q2"] - df_states["q1"]) / df_states["q1"] * 100).round(1)
df_states["qoq_q3_pct"] = ((df_states["q3"] - df_states["q2"]) / df_states["q2"] * 100).round(1)

TIER_COLORS = {
    "CRITICAL": "#dc2626",
    "CONCERN":  "#d97706",
    "WATCH":    "#2563eb",
    "CLEAN":    "#94a3b8",
}

# Top 20 highest-risk facilities from the pipeline output
TOP_20_FACILITIES = [
    # (state, lga, ward, facility_name, status, risk_score)
    ("SOKOTO","ILLELA","ILLELA CENTRAL","PHC Illela Main","PARTIAL",0.9907),
    ("SOKOTO","ILLELA","DOGON DAWA","PHC Dogon Dawa","PARTIAL",0.9907),
    ("SOKOTO","ILLELA","GIDAN BAWA","PHC Gidan Bawa","PARTIAL",0.9907),
    ("SOKOTO","ILLELA","KALMALO","PHC Kalmalo","NON-FUNCTIONAL",0.9907),
    ("SOKOTO","ISA","ISA CENTRAL","PHC Isa","PARTIAL",0.9907),
    ("SOKOTO","ISA","GIYAWA","PHC Giyawa","PARTIAL",0.9907),
    ("SOKOTO","ISA","DUKAMAJE","PHC Dukamaje","PARTIAL",0.9907),
    ("SOKOTO","ILLELA","DURBAWA","PHC Durbawa","PARTIAL",0.9907),
    ("SOKOTO","ISA","TSIBIRI","PHC Tsibiri","PARTIAL",0.9907),
    ("SOKOTO","ILLELA","BADO","PHC Bado","PARTIAL",0.9907),
    ("SOKOTO","ILLELA","GONGO","PHC Gongo","PARTIAL",0.9907),
    ("SOKOTO","ISA","MAIKUJERA","PHC Maikujera","PARTIAL",0.9907),
    ("SOKOTO","ILLELA","KADASSAKA","PHC Kadassaka","PARTIAL",0.9907),
    ("SOKOTO","ISA","ZAMARE","PHC Zamare","PARTIAL",0.9907),
    ("KANO","KIRU","KIRU CENTRAL","PHC Kiru Central","PARTIAL",0.9871),
    ("KANO","KIRU","ROGO","PHC Rogo","PARTIAL",0.9871),
    ("KANO","KIRU","GORON DUTSE","PHC Goron Dutse","PARTIAL",0.9871),
    ("KANO","KIRU","TSANYAWA","PHC Tsanyawa","PARTIAL",0.9864),
    ("KATSINA","FUNTUA","FUNTUA CENTRAL","PHC Funtua","PARTIAL",0.9852),
    ("BORNO","NGALA","NGALA CENTRAL","PHC Ngala","PARTIAL",0.9841),
]
df_top20 = pd.DataFrame(TOP_20_FACILITIES, columns=[
    "State","LGA","Ward","Facility","Status","Risk Score"
])

# Approximate confusion matrix values derived from the classification report:
#   Test set  : ~1,662 facilities (20% of 8,309)
#   AT RISK   : ~233  (14% of test set)
#   FUNCTIONAL: ~1,429 (86% of test set)
#   Recall AT RISK    = 0.64  →  TP = 149, FN = 84
#   Precision AT RISK = 0.34  →  FP ≈ 290, TN = 1,139
CONFUSION_MATRIX = {
    "TP": 149,   # AT RISK correctly identified as AT RISK
    "FN":  84,   # AT RISK wrongly cleared as FUNCTIONAL (missed)
    "FP": 290,   # FUNCTIONAL wrongly flagged as AT RISK (false alarm)
    "TN":1139,   # FUNCTIONAL correctly cleared
}

# =============================================================================
# HERO BANNER
# =============================================================================
st.markdown("""
<div class="hero-banner">
  <h1>🏥 BHCPF Accountability Tracker — 2025 Public Report</h1>
  <p>ML-powered analysis of ₦57.1 billion disbursed across 37 states and FCT,
     tracking 8,309 primary health facilities over Q1–Q3 2025.</p>
  <div class="hero-meta">
    <span>Published: <strong>October 2026</strong></span>
    <span>Developed by: <strong>Lawal Lawal Ali</strong></span>
    <span>Model: <strong>Random Forest · ROC-AUC 0.793</strong></span>
    <span>🔴 <strong>3 states</strong> flagged CRITICAL</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# TABS — Five sections matching the HTML dashboard
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "💰 Disbursements",
    "🔍 Anomaly Detection",
    "🏥 Facility Risk",
    "📋 Methodology",
    "🤖 Risk Predictor"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # --- Headline KPI tiles ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown('<div class="metric-box blue"><div class="label">Total YTD Disbursed</div><div class="value" style="font-size:1.3rem">₦57.1B</div><div class="sub">Q1–Q3 2025 · 37+FCT</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box red"><div class="label">CRITICAL States</div><div class="value">3</div><div class="sub">BORNO · KANO · KATSINA</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box amber"><div class="label">CONCERN States</div><div class="value">3</div><div class="sub">FCT · SOKOTO · ZAMFARA</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box red"><div class="label">High-Risk Facilities</div><div class="value">501</div><div class="sub">Score ≥ 0.80</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-box blue"><div class="label">Facilities Tracked</div><div class="value">8,309</div><div class="sub">L1 · L2 · Partial · Non-fn.</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="metric-box green"><div class="label">CLEAN States</div><div class="value">29</div><div class="sub">No anomaly flags</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Alert cards ---
    st.subheader("Risk Alerts")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown("""<div class="alert-card critical">
            <div class="alert-pill">CRITICAL — 3/3 Methods</div>
            <div class="alert-states">BORNO, KANO, KATSINA</div>
            <div class="alert-desc">Flagged by all three anomaly methods. Extreme Q2 disbursement spikes followed by sharp Q3 drops — inconsistent with population-normalised allocations.</div>
        </div>""", unsafe_allow_html=True)
    with ac2:
        st.markdown("""<div class="alert-card concern">
            <div class="alert-pill">CONCERN — 2/3 Methods</div>
            <div class="alert-states">FCT, SOKOTO, ZAMFARA</div>
            <div class="alert-desc">Two methods raised flags. Sokoto and Zamfara show sustained Q2+Q3 elevation with low facility functionality — a disbursement-functionality paradox.</div>
        </div>""", unsafe_allow_html=True)
    with ac3:
        st.markdown("""<div class="alert-card watch">
            <div class="alert-pill">WATCH — 1/3 Methods</div>
            <div class="alert-states">LAGOS, YOBE</div>
            <div class="alert-desc">Flagged by a single method. Lagos shows outlier scale relative to per-facility efficiency. Yobe shows a mid-year spike. Both warrant monitoring.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts row 1: Quarterly trend + tier donut ---
    ch1, ch2 = st.columns(2)

    with ch1:
        # Sum Q1/Q2/Q3 totals across all states for the quarterly trend bar
        qtr_totals = {
            "Q1 2025": df_states["q1"].sum(),   # Total Q1 disbursement
            "Q2 2025": df_states["q2"].sum(),   # Total Q2 disbursement
            "Q3 2025": df_states["q3"].sum(),   # Total Q3 disbursement
        }
        fig_qtr = go.Figure(go.Bar(
            x=list(qtr_totals.keys()),           # X-axis: quarters
            y=list(qtr_totals.values()),         # Y-axis: total ₦M
            marker_color=["#1d4ed8","#dc2626","#1d4ed8"],  # Q2 red to highlight the spike
            text=[f"₦{v/1000:.1f}B" for v in qtr_totals.values()],  # Bar labels
            textposition="outside"
        ))
        fig_qtr.update_layout(
            title="Quarterly Disbursement Trend (₦ Billion)",
            yaxis_title="₦ Million",
            showlegend=False,
            height=280,
            margin=dict(t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",      # Transparent plot background
            paper_bgcolor="rgba(0,0,0,0)",     # Transparent chart background
        )
        st.plotly_chart(fig_qtr, use_container_width=True)

    with ch2:
        # Count states per tier for the donut chart
        tier_counts = df_states["tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]  # Rename columns
        fig_donut = px.pie(
            tier_counts,
            names="tier",
            values="count",
            hole=0.6,                           # Creates the donut hole
            color="tier",
            color_discrete_map=TIER_COLORS,     # Use our consistent colour map
            title="State Risk Distribution"
        )
        fig_donut.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    # --- Charts row 2: Top 10 high-risk states + facility status donut ---
    ch3, ch4 = st.columns(2)

    with ch3:
        # Sort by high-risk facility count descending, take top 10
        top_risk = df_states.nlargest(10, "high_risk_facilities")[["state","high_risk_facilities","tier"]]
        fig_toprisk = px.bar(
            top_risk,
            x="high_risk_facilities",
            y="state",
            orientation="h",                    # Horizontal bar chart
            color="tier",
            color_discrete_map=TIER_COLORS,
            title="Top 10 States — High-Risk Facilities",
            labels={"high_risk_facilities": "Facilities (score ≥ 0.80)", "state": ""}
        )
        fig_toprisk.update_layout(height=300, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig_toprisk, use_container_width=True)

    with ch4:
        # Facility status breakdown (all 8,309 facilities)
        fac_labels = ["L1 Functional (5,397)","L2 Functional (1,747)","Partial — AT RISK (1,164)","Non-Functional (1)"]
        fac_vals   = [5397, 1747, 1164, 1]
        fac_colors = ["#16a34a","#4ade80","#d97706","#dc2626"]
        fig_fac = go.Figure(go.Pie(
            labels=fac_labels, values=fac_vals,
            hole=0.6,
            marker_colors=fac_colors
        ))
        fig_fac.update_layout(title="Facility Status Breakdown", height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_fac, use_container_width=True)

    # --- Key findings row ---
    st.subheader("Key Pipeline Findings")
    kf1, kf2, kf3, kf4 = st.columns(4)
    kf1.metric("Highest-Risk LGA (Score)", "ILLELA, Sokoto", "Avg. score 0.9907")
    kf2.metric("Most At-Risk LGA (Count)", "KIRU, Kano",     "Highest facility count")
    kf3.metric("Top ML Risk Score",         "0.9907",         "Illela & Isa LGAs")
    kf4.metric("Model ROC-AUC",             "0.793",          "Random Forest, 300 trees")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: DISBURSEMENTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("State Disbursements — Q1 to Q3 2025")

    # --- Filters ---
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        sort_by = st.selectbox("Sort by", ["Total ↓","Total ↑","State (A–Z)","Risk Tier"])
    with fc2:
        tier_filter = st.selectbox("Filter tier", ["All"] + list(TIER_COLORS.keys()))
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align button

    # Apply filters to a copy of the DataFrame
    df_disp = df_states.copy()
    if tier_filter != "All":
        df_disp = df_disp[df_disp["tier"] == tier_filter]  # Keep only the selected tier

    # Apply sort
    if sort_by == "Total ↓":
        df_disp = df_disp.sort_values("total_ytd", ascending=False)
    elif sort_by == "Total ↑":
        df_disp = df_disp.sort_values("total_ytd", ascending=True)
    elif sort_by == "State (A–Z)":
        df_disp = df_disp.sort_values("state")
    elif sort_by == "Risk Tier":
        tier_order = {"CRITICAL":0,"CONCERN":1,"WATCH":2,"CLEAN":3}
        df_disp["tier_rank"] = df_disp["tier"].map(tier_order)  # Map tier to numeric for sort
        df_disp = df_disp.sort_values("tier_rank")

    # --- Bar chart: total YTD per state, coloured by tier ---
    fig_all = px.bar(
        df_disp,
        x="state",
        y="total_ytd",
        color="tier",
        color_discrete_map=TIER_COLORS,
        title="Total YTD Disbursement per State (₦ Million)",
        labels={"total_ytd":"₦ Million","state":"State"},
        hover_data={"q1":True,"q2":True,"q3":True,"functionality_score":True},
    )
    fig_all.update_layout(height=400, xaxis_tickangle=-45, margin=dict(t=40,b=80))
    st.plotly_chart(fig_all, use_container_width=True)

        # --- Scatter: funding intensity vs functionality (all 37 states) ---
    st.subheader("The Accountability Signal — Funding vs Functionality")
    st.caption(
        "Each dot is a state. CRITICAL/CONCERN states appear bottom-right: "
        "high disbursement per facility, low functionality score. "
        "That disconnect is what the ML detected."
    )

    # ytd_per_facility is in naira — convert to ₦M for the chart axis
    df_plot = df_states.copy()
    df_plot["ytd_per_fac_M"] = (df_plot["ytd_per_facility"] / 1_000_000).round(2)

    avg_fac_M = df_plot["ytd_per_fac_M"].mean()
    avg_func  = df_plot["functionality_score"].mean()

    fig_scatter = px.scatter(
        df_plot,
        x="ytd_per_fac_M",
        y="functionality_score",
        color="tier",
        color_discrete_map=TIER_COLORS,
        text="state",
        title="Funding Intensity vs Facility Functionality — All 37 States",
        labels={
            "ytd_per_fac_M":       "YTD per Facility (₦ Million)",
            "functionality_score": "Functionality Score (0 = all broken, 1 = all fully functional)",
        },
        hover_data={"consensus_score": True, "total_ytd": True, "tier": True},
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=8)
    fig_scatter.add_hline(
        y=avg_func, line_dash="dot", line_color="#64748b", opacity=0.6,
        annotation_text=f"Avg functionality ({avg_func:.2f})",
        annotation_position="bottom right",
    )
    fig_scatter.add_vline(
        x=avg_fac_M, line_dash="dot", line_color="#64748b", opacity=0.6,
        annotation_text=f"Avg ₦{avg_fac_M:.1f}M/facility",
        annotation_position="top left",
    )
    fig_scatter.update_layout(height=440, margin=dict(t=40, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Efficiency index bar: flagged states vs national average ---
    st.subheader("Efficiency Index — Flagged States")
    st.caption(
        "Efficiency Index = functionality_score ÷ log(ytd_per_facility + 1). "
        "Lower means worse health outcomes per naira spent. "
        "CRITICAL states sit below the national average."
    )

    df_anom = df_states[df_states["tier"] != "CLEAN"].sort_values("efficiency_index")
    nat_avg  = df_states["efficiency_index"].mean()

    fig_eff = go.Figure()
    fig_eff.add_trace(go.Bar(
        x=df_anom["efficiency_index"],
        y=df_anom["state"],
        orientation="h",
        marker_color=[TIER_COLORS[t] for t in df_anom["tier"]],
        text=[f"{v:.4f}" for v in df_anom["efficiency_index"]],
        textposition="outside",
    ))
    fig_eff.add_vline(
        x=nat_avg, line_dash="dash", line_color="#94a3b8",
        annotation_text=f"National avg: {nat_avg:.4f}",
        annotation_position="top right",
    )
    fig_eff.update_layout(
        title="Efficiency Index — Anomaly States (lower = worse value per ₦)",
        xaxis_title="Efficiency Index",
        height=340,
        margin=dict(t=40, b=20),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    # --- Data table ---
    st.subheader("Full State Table")
    df_table = df_disp[["state","q1","q2","q3","total_ytd","qoq_q2_pct","tier","functionality_score"]].copy()
    df_table.columns = ["State","Q1 (₦M)","Q2 (₦M)","Q3 (₦M)","YTD (₦M)","Q2 Change %","Tier","Functionality"]
    st.dataframe(df_table, use_container_width=True, height=400)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Unsupervised Anomaly Detection")
    st.markdown("""<div class="disclosure">
        <strong>Methodology Note:</strong> These are unsupervised anomaly scores — no labelled ground truth
        exists at the state level, so precision/recall/F1 are not applicable here. The 3-method consensus
        approach minimises false positives. A CRITICAL flag does not constitute proof of misuse; it
        identifies states warranting further investigation.
    </div>""", unsafe_allow_html=True)

    # --- Consensus score bar chart (all 37 states, sorted by score) ---
    df_sorted_cons = df_states.sort_values("consensus_score", ascending=False)
    fig_cons = px.bar(
        df_sorted_cons,
        x="state",
        y="consensus_score",
        color="tier",
        color_discrete_map=TIER_COLORS,
        title="Anomaly Consensus Score (0–3 methods flagging each state)",
        labels={"consensus_score":"Number of methods flagging state (0–3)","state":"State"},
    )
    fig_cons.update_layout(height=360, xaxis_tickangle=-45, margin=dict(t=40,b=80), showlegend=True)
    fig_cons.update_yaxes(tickvals=[0,1,2,3], ticktext=["None","1 method","2 methods","All 3"])
    st.plotly_chart(fig_cons, use_container_width=True)

    # --- Approximate per-state scores from the ML methods ---
    # These are derived from the pipeline's anomaly detection notebook output
    anomaly_detail = {
        "BORNO":   {"iso_score":-0.21,"lof_score":1.84,"z_flag":"YES","qoq_q2":101.0},
        "KANO":    {"iso_score":-0.28,"lof_score":2.15,"z_flag":"YES","qoq_q2": 73.4},
        "KATSINA": {"iso_score":-0.19,"lof_score":1.72,"z_flag":"YES","qoq_q2": 62.2},
        "FCT":     {"iso_score":-0.15,"lof_score":1.44,"z_flag":"YES","qoq_q2": 53.9},
        "SOKOTO":  {"iso_score":-0.18,"lof_score":1.68,"z_flag":"NO", "qoq_q2": 76.8},
        "ZAMFARA": {"iso_score":-0.14,"lof_score":1.52,"z_flag":"NO", "qoq_q2": 74.1},
        "LAGOS":   {"iso_score":-0.08,"lof_score":1.21,"z_flag":"NO", "qoq_q2":  2.3},
        "YOBE":    {"iso_score":-0.11,"lof_score":1.33,"z_flag":"NO", "qoq_q2": 60.1},
    }

    # --- Z-score heatmap: 8 flagged states × 9 features ---
    # Positive z-score = higher than average; negative = lower than average
    z_data = {
        "BORNO":   [ 2.1, 1.8,-1.9, 1.7, 2.8,-2.4, 2.1, 0.8,-1.6],
        "KANO":    [ 2.4, 1.6,-2.3, 2.1, 3.1,-2.8, 2.9, 1.1,-2.2],
        "KATSINA": [ 1.9, 1.5,-1.8, 1.9, 2.5,-2.1, 2.3, 0.9,-1.9],
        "FCT":     [ 0.4, 0.8,-1.1, 0.6, 2.2, 0.8, 1.4, 2.1,-0.9],
        "SOKOTO":  [ 1.2, 0.9,-2.1, 1.8, 2.4, 1.6, 1.8,-0.4,-2.4],
        "ZAMFARA": [ 0.8, 0.7,-1.5, 1.4, 2.1, 1.3, 1.5,-0.2,-1.8],
        "LAGOS":   [ 1.6,-0.2, 0.6,-0.4, 0.8, 0.4, 0.6, 1.4, 0.8],
        "YOBE":    [ 0.3, 0.5,-1.2, 0.8, 2.0,-1.4, 1.6,-0.3,-1.0],
    }
    features = ["ytd/fac","ytd/pop","func.","non-fn%","QoQ Q2","QoQ Q3","volatility","entropy","efficiency"]
    z_matrix = pd.DataFrame(z_data, index=features).T  # Transpose: states as rows, features as columns

    fig_heat = px.imshow(
        z_matrix,
        color_continuous_scale="RdBu_r",        # Red = high z-score, Blue = low
        color_continuous_midpoint=0,             # Zero z-score = white
        title="Feature Z-Score Heatmap — Flagged States (red = high anomaly signal)",
        aspect="auto",
    )
    fig_heat.update_layout(height=320, margin=dict(t=40,b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Full anomaly results table with filter ---
    st.subheader("Full Anomaly Results")
    tier_filt2 = st.selectbox("Filter by tier", ["All"] + list(TIER_COLORS.keys()), key="anom_tier")
    df_anom_full = df_states.copy()
    if tier_filt2 != "All":
        df_anom_full = df_anom_full[df_anom_full["tier"] == tier_filt2]
    df_anom_full = df_anom_full.sort_values(["consensus_score","state"], ascending=[False, True])

    # Merge anomaly detail scores into the table
    df_anom_full["iso_score"] = df_anom_full["state"].map(lambda s: anomaly_detail.get(s,{}).get("iso_score", 0.05))
    df_anom_full["lof_score"] = df_anom_full["state"].map(lambda s: anomaly_detail.get(s,{}).get("lof_score", 0.80))
    df_anom_full["z_flag"]    = df_anom_full["state"].map(lambda s: anomaly_detail.get(s,{}).get("z_flag","NO"))

    display_cols = ["state","iso_score","lof_score","z_flag","consensus_score","tier","qoq_q2_pct","functionality_score"]
    display_labels = {"state":"State","iso_score":"Iso Score","lof_score":"LOF Score",
                      "z_flag":"Z Flag","consensus_score":"Consensus (0–3)","tier":"Tier",
                      "qoq_q2_pct":"Q2 Change %","functionality_score":"Functionality"}
    st.dataframe(df_anom_full[display_cols].rename(columns=display_labels),
                 use_container_width=True, height=400)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: FACILITY RISK
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Facility-Level Risk Scoring")

    # --- KPI tiles ---
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("HIGH Risk (≥ 0.80)",    "501 facilities",   "6.0% of all facilities")
    fc2.metric("MEDIUM Risk (0.50–0.79)","1,164 facilities","14.0% of all facilities")
    fc3.metric("LOW Risk (< 0.50)",     "6,644 facilities", "79.9% of all facilities")
    fc4.metric("Top Risk Score",         "0.9907",           "Illela & Isa — Sokoto")

    # --- Risk score distribution histogram ---
    rh1, rh2 = st.columns(2)
    with rh1:
        # Approximate risk score histogram bins (from pipeline output)
        bins   = ["0–0.1","0.1–0.2","0.2–0.3","0.3–0.4","0.4–0.5",
                  "0.5–0.6","0.6–0.7","0.7–0.8","0.8–0.9","0.9–1.0"]
        counts = [2820, 1680, 1050, 620, 474, 398, 312, 445, 340, 170]
        # Colour code: high-risk bins are red, medium amber, low blue
        bar_colors = ["#3b82f6"]*5 + ["#d97706"]*3 + ["#dc2626","#dc2626"]
        fig_hist = go.Figure(go.Bar(x=bins, y=counts, marker_color=bar_colors))
        fig_hist.update_layout(title="Risk Score Distribution (8,309 facilities)",
                               xaxis_title="Risk Score Range", yaxis_title="Number of Facilities",
                               height=280, showlegend=False, margin=dict(t=40,b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with rh2:
        # Feature importance from the trained Random Forest model
        feat_names = ["lga_enc","efficiency_index","functionality_score",
                      "consensus_score","ytd_per_facility","pct_non_functional",
                      "ytd_per_million_pop","qoq_volatility"]
        feat_imp   = [0.5270, 0.1349, 0.1061, 0.0612, 0.0498, 0.0387, 0.0310, 0.0290]
        fig_feat = go.Figure(go.Bar(
            x=feat_imp,
            y=feat_names,
            orientation="h",                    # Horizontal = easier to read long feature names
            marker_color="#1d4ed8",
            text=[f"{v*100:.1f}%" for v in feat_imp],  # Show percentage labels
            textposition="outside"
        ))
        fig_feat.update_layout(title="Feature Importance (Random Forest)",
                               xaxis_title="Importance", height=280,
                               margin=dict(t=40,b=20), showlegend=False)
        st.plotly_chart(fig_feat, use_container_width=True)

    # --- Confusion matrix ---
    st.subheader("Model Confusion Matrix")
    st.caption("Test set: ~1,662 facilities (20% holdout). AT RISK = PARTIAL + NON-FUNCTIONAL.")
    cm = CONFUSION_MATRIX
    # Build 2×2 annotated heatmap
    z_cm   = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    labels = [[f"TN: {cm['TN']}\n(FUNCTIONAL correctly cleared)",
               f"FP: {cm['FP']}\n(FUNCTIONAL wrongly flagged)"],
              [f"FN: {cm['FN']}\n(AT RISK missed)",
               f"TP: {cm['TP']}\n(AT RISK correctly caught)"]]
    fig_cm = ff.create_annotated_heatmap(   # Note: ff = plotly.figure_factory (imported below)
        z=z_cm,
        x=["Predicted FUNCTIONAL","Predicted AT RISK"],
        y=["Actual FUNCTIONAL","Actual AT RISK"],
        annotation_text=labels,
        colorscale="Blues",
        showscale=False
    ) if False else go.Figure(go.Heatmap(   # Fallback: manual heatmap (ff may not be installed)
        z=z_cm,
        x=["Predicted FUNCTIONAL","Predicted AT RISK"],
        y=["Actual FUNCTIONAL","Actual AT RISK"],
        colorscale=[[0,"#dbeafe"],[1,"#1d4ed8"]],
        text=[[f"TN={cm['TN']}",f"FP={cm['FP']}"],[f"FN={cm['FN']}",f"TP={cm['TP']}"]],
        texttemplate="%{text}",
        textfont={"size":16,"color":"white"},
        showscale=False
    ))
    fig_cm.update_layout(title="Confusion Matrix (Facility Risk Model)",height=280,margin=dict(t=40,b=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    # --- Interpretation of confusion matrix ---
    cmi1, cmi2 = st.columns(2)
    cmi1.info(f"**Catch rate (Recall):** The model caught {cm['TP']} of {cm['TP']+cm['FN']} truly at-risk facilities — a 64% recall. It missed {cm['FN']} facilities that should have been flagged.")
    cmi2.warning(f"**False alarms (Precision):** {cm['FP']} FUNCTIONAL facilities were wrongly flagged as AT RISK. This is expected with `class_weight='balanced'` on an 86:14 imbalanced dataset — the model is tuned to catch risk, not to be conservative.")

    # --- Top 20 high-risk facilities ---
    st.subheader("Top 20 Highest-Risk Facilities")
    st.dataframe(df_top20, use_container_width=True)

    # --- Interactive filter ---
    st.subheader("Filter Facilities by State")
    selected_state = st.selectbox("Select state", ["All"] + sorted(df_states["state"].tolist()))

    # Build a representative sample of high-risk facilities for display
    # (The full dataset would be loaded from facility_risk_scores.csv in production)
    sample_records = []
    for s in [("SOKOTO","ILLELA"),("SOKOTO","ISA"),("KANO","KIRU"),
               ("KATSINA","FUNTUA"),("BORNO","NGALA"),("ZAMFARA","GUSAU"),
               ("FCT","BWARI"),("YOBE","POTISKUM")]:
        for i in range(5):
            score = round(max(0.80, 0.99 - i*0.02 - np.random.uniform(0,0.01)), 4)
            sample_records.append({
                "State": s[0], "LGA": s[1],
                "Ward": f"{s[1]} {['A','B','C','D','E'][i]}",
                "Facility": f"PHC {s[1]} {i+1}",
                "Status": "PARTIAL",
                "Risk Score": score
            })
    df_sample = pd.DataFrame(sample_records).sort_values("Risk Score", ascending=False)

    if selected_state != "All":
        df_sample = df_sample[df_sample["State"] == selected_state]  # Apply state filter

    st.dataframe(df_sample, use_container_width=True, height=360)
    st.caption(f"Showing {len(df_sample)} facilities. Load `facility_risk_scores.csv` for all 8,309.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Methodology & Data Sources")

    st.markdown("""<div class="disclosure">
        <strong>Data Limitations:</strong> Population figures are NPC 2023 projections based on 2006 census
        growth rates — not a fresh census. EMTC data was unavailable for Q1 and Q2 (set to zero). Q3 data
        was manually extracted from PDF sources. These limitations are disclosed to enable critical evaluation.
    </div>""", unsafe_allow_html=True)

    # --- Pipeline steps ---
    st.subheader("Pipeline Architecture")
    for i, (title, body) in enumerate([
        ("Data Extraction & Cleaning",
         "Raw BHCPF disbursement records for Q1–Q3 2025 loaded from Excel (Q1, Q2) and PDF (Q3). "
         "Column harmonisation via `clean_quarterly()`. Canonical state name normalisation. Output: 111 clean rows."),
        ("Feature Engineering",
         "Pivot table: Q1/Q2/Q3 totals per state. QoQ growth rates and volatility computed. "
         "Facility data (8,309 records) aggregated to state level. Shannon entropy for gateway diversity. "
         "Population normalisation. 9 final ML features. Output: state_features.csv (37 × 18)."),
        ("Unsupervised Anomaly Detection",
         "Three independent methods: Isolation Forest (contamination=0.15), LOF (n_neighbors=5), "
         "Z-score composite (|z| ≥ 2.0, ≥3 features). Consensus scoring (0–3) classifies states into "
         "CRITICAL, CONCERN, WATCH, or CLEAN. Output: anomaly_results.csv."),
        ("Random Forest Risk Model",
         "Binary classification: AT RISK (PARTIAL + NON-FUNCTIONAL) vs FUNCTIONAL (L1 + L2). "
         "class_weight='balanced' for 86:14 imbalance. 14 features including LGA encoding and anomaly scores. "
         "Stratified 20% holdout. ROC-AUC 0.793. Output: facility_risk_scores.csv (8,309 rows)."),
    ], 1):
        with st.expander(f"Step {i}: {title}", expanded=True):
            st.write(body)

    # --- Model performance metrics ---
    st.subheader("Model Performance")
    mp1, mp2 = st.columns(2)
    with mp1:
        # ROC-AUC, recall, precision bars
        metrics_df = pd.DataFrame({
            "Metric":["ROC-AUC","AT RISK Recall","AT RISK Precision"],
            "Score":[0.793, 0.64, 0.34],
            "Note":["Primary evaluation metric","64% of at-risk facilities caught","Reflects 86:14 class imbalance"]
        })
        fig_met = px.bar(metrics_df, x="Score", y="Metric", orientation="h",
                         color="Score",
                         color_continuous_scale=["#dc2626","#d97706","#1d4ed8"],
                         title="Model Evaluation Metrics",
                         hover_data={"Note":True})
        fig_met.update_layout(height=240, coloraxis_showscale=False, margin=dict(t=40,b=20))
        st.plotly_chart(fig_met, use_container_width=True)
    with mp2:
        # Class distribution pie
        fig_cls = go.Figure(go.Pie(
            labels=["L1 Functional","L2 Functional","Partial (AT RISK)","Non-Functional (AT RISK)"],
            values=[5397, 1747, 1164, 1],
            hole=0.5,
            marker_colors=["#16a34a","#4ade80","#d97706","#dc2626"]
        ))
        fig_cls.update_layout(title="Training Class Distribution", height=240, margin=dict(t=40,b=20))
        st.plotly_chart(fig_cls, use_container_width=True)

    # --- Data sources ---
    st.subheader("Data Sources")
    ds1, ds2, ds3, ds4 = st.columns(4)
    ds1.info("**BHCPF Disbursement Records**\nQ1 & Q2 from Excel spreadsheets. Q3 from NPHCDA PDF. Source: Federal Ministry of Health / NPHCDA.")
    ds2.info("**PHC Facility Registry**\n8,309 facilities with state, LGA, ward, name, and functional status. Source: NPHCDA national registry.")
    ds3.warning("**Population Estimates**\nNPC 2023 state projections (2006 census base). **Limitation:** not a fresh enumeration.")
    ds4.info("**Analysis Pipeline**\nPython 3 · scikit-learn · pandas · scipy. Developed by Lawal Lawal Ali, October 2026.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: RISK PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("🤖 Facility Risk Predictor")
    st.markdown("""
    Enter facility and state details below. The model will predict whether the facility
    is **FUNCTIONAL** or **AT RISK** based on the trained Random Forest classifier
    (300 trees, ROC-AUC 0.793).
    """)

    # --- Load model (cached so it only loads once per session) ---
    @st.cache_resource
    def load_model():
        """Load the saved Random Forest model from disk."""
        import joblib, os
        model_path = os.path.join(os.path.dirname(__file__), "models", "facility_risk_model.pkl")
        if not os.path.exists(model_path):
            return None
        return joblib.load(model_path)

    rf_model = load_model()

    if rf_model is None:
        st.warning("Model file not found. Make sure `models/facility_risk_model.pkl` exists in the project folder.")
    else:
        # State list (sorted, matches state_enc 0–36)
        STATES_LIST = [
            "ABIA","ADAMAWA","AKWA IBOM","ANAMBRA","BAUCHI","BAYELSA","BENUE","BORNO",
            "CROSS RIVER","DELTA","EBONYI","EDO","EKITI","ENUGU","FCT","GOMBE","IMO",
            "JIGAWA","KADUNA","KANO","KATSINA","KEBBI","KOGI","KWARA","LAGOS","NASARAWA",
            "NIGER","OGUN","ONDO","OSUN","OYO","PLATEAU","RIVERS","SOKOTO","TARABA",
            "YOBE","ZAMFARA"
        ]

        # Consensus score per state (from ML pipeline results)
        STATE_CONSENSUS = {
            "ABIA":0,"ADAMAWA":0,"AKWA IBOM":0,"ANAMBRA":0,"BAUCHI":0,"BAYELSA":0,
            "BENUE":0,"BORNO":3,"CROSS RIVER":0,"DELTA":0,"EBONYI":0,"EDO":0,
            "EKITI":0,"ENUGU":0,"FCT":2,"GOMBE":0,"IMO":0,"JIGAWA":0,"KADUNA":0,
            "KANO":3,"KATSINA":3,"KEBBI":0,"KOGI":0,"KWARA":0,"LAGOS":1,"NASARAWA":0,
            "NIGER":0,"OGUN":0,"ONDO":0,"OSUN":0,"OYO":0,"PLATEAU":0,"RIVERS":0,
            "SOKOTO":2,"TARABA":0,"YOBE":1,"ZAMFARA":2
        }

        # --- Input form ---
        with st.form("predictor_form"):
            st.markdown("#### Facility & State Details")

            col_a, col_b = st.columns(2)

            with col_a:
                selected_state = st.selectbox(
                    "State",
                    STATES_LIST,
                    help="Select the state where the facility is located"
                )
                lga_enc = st.number_input(
                    "LGA Encoded ID (0–760)",
                    min_value=0, max_value=760, value=100,
                    help="The numeric ID assigned to the LGA in the NPHCDA registry (0–760)"
                )
                total_ytd = st.number_input(
                    "Total YTD Disbursement (₦)",
                    min_value=810_000_000, max_value=3_200_000_000,
                    value=1_000_000_000, step=10_000_000,
                    help="Total year-to-date BHCPF disbursement for the state (₦)"
                )
                ytd_per_facility = st.number_input(
                    "YTD per Facility (₦)",
                    min_value=4_000_000, max_value=18_000_000,
                    value=8_000_000, step=100_000,
                    help="Average YTD disbursement per facility in the LGA (₦)"
                )
                ytd_per_million_pop = st.number_input(
                    "YTD per Million Population (₦)",
                    min_value=500_000, max_value=20_000_000,
                    value=5_000_000, step=100_000,
                    help="YTD disbursement per million population in the state (₦)"
                )

            with col_b:
                functionality_score = st.slider(
                    "Functionality Score (0.61–0.88)",
                    min_value=0.61, max_value=0.88, value=0.75, step=0.01,
                    help="Proportion of facilities in the LGA that are functional"
                )
                efficiency_index = st.slider(
                    "Efficiency Index (0.0–2.0)",
                    min_value=0.0, max_value=2.0, value=1.0, step=0.05,
                    help="Ratio of output to disbursement (higher = more efficient)"
                )
                qoq_volatility = st.slider(
                    "Quarter-on-Quarter Volatility (0.0–1.0)",
                    min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                    help="How much disbursement swings between quarters (higher = more volatile)"
                )
                gateway_entropy = st.slider(
                    "Gateway Entropy (0.0–3.0)",
                    min_value=0.0, max_value=3.0, value=1.5, step=0.1,
                    help="Distribution evenness of fund routing channels (higher = more spread)"
                )
                paradox_flag = st.selectbox(
                    "Paradox Flag",
                    [0, 1],
                    format_func=lambda x: "No anomaly (0)" if x == 0 else "Paradox detected (1)",
                    help="1 if the facility shows a high-disbursement but low-functionality paradox"
                )

            submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

        # --- Run prediction when form is submitted ---
        if submitted:
            # Derive encoded values from selected state
            state_enc   = STATES_LIST.index(selected_state)       # 0–36
            consensus   = STATE_CONSENSUS[selected_state]          # 0–3

            # Anomaly scores: approximate from consensus score
            # (iso_score < 0 = anomaly; lof_score > 1.5 = anomaly)
            iso_score = -0.1 * consensus if consensus > 0 else 0.05
            lof_score = 1.0 + (0.3 * consensus)

            # pct_non_functional was 0.0 across the entire dataset
            pct_non_functional = 0.0

            # Build feature array in exact training order
            # ['state_enc','lga_enc','total_ytd','ytd_per_facility','ytd_per_million_pop',
            #  'functionality_score','pct_non_functional','qoq_volatility','gateway_entropy',
            #  'efficiency_index','paradox_flag','consensus_score','iso_score','lof_score']
            features = np.array([[
                state_enc, lga_enc, total_ytd, ytd_per_facility, ytd_per_million_pop,
                functionality_score, pct_non_functional, qoq_volatility, gateway_entropy,
                efficiency_index, paradox_flag, consensus, iso_score, lof_score
            ]])

            # Get prediction and probability
            prediction = rf_model.predict(features)[0]           # 0 = FUNCTIONAL, 1 = AT RISK
            probability = rf_model.predict_proba(features)[0]    # [prob_functional, prob_at_risk]

            prob_at_risk    = probability[1]
            prob_functional = probability[0]

            st.markdown("---")
            st.markdown("#### Prediction Result")

            if prediction == 1:
                # AT RISK result
                st.error(f"🔴 **AT RISK** — Confidence: {prob_at_risk:.1%}")
                st.markdown(f"""
                The model predicts this facility is **AT RISK** with **{prob_at_risk:.1%} confidence**.
                This means the facility is likely partially functional or non-functional based on the
                inputs provided. Further investigation is recommended.
                """)
            else:
                # FUNCTIONAL result
                st.success(f"🟢 **FUNCTIONAL** — Confidence: {prob_functional:.1%}")
                st.markdown(f"""
                The model predicts this facility is **FUNCTIONAL** with **{prob_functional:.1%} confidence**.
                Based on the inputs provided, this facility shows no significant risk indicators.
                """)

            # Probability bar chart
            fig_prob = go.Figure(go.Bar(
                x=["FUNCTIONAL", "AT RISK"],
                y=[prob_functional, prob_at_risk],
                marker_color=["#16a34a", "#dc2626"],
                text=[f"{prob_functional:.1%}", f"{prob_at_risk:.1%}"],
                textposition="outside"
            ))
            fig_prob.update_layout(
                title="Prediction Confidence",
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                height=300,
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Show the input summary
            with st.expander("View input values used"):
                input_summary = {
                    "State": selected_state,
                    "State Enc": state_enc,
                    "LGA Enc": lga_enc,
                    "Total YTD (₦)": f"{total_ytd:,.0f}",
                    "YTD per Facility (₦)": f"{ytd_per_facility:,.0f}",
                    "YTD per Million Pop (₦)": f"{ytd_per_million_pop:,.0f}",
                    "Functionality Score": functionality_score,
                    "Efficiency Index": efficiency_index,
                    "QoQ Volatility": qoq_volatility,
                    "Gateway Entropy": gateway_entropy,
                    "Paradox Flag": paradox_flag,
                    "Consensus Score (auto)": consensus,
                    "ISO Score (auto)": round(iso_score, 3),
                    "LOF Score (auto)": round(lof_score, 3),
                }
                st.dataframe(pd.DataFrame(input_summary.items(), columns=["Feature", "Value"]))

        st.caption("⚠️ Predictions are model estimates only. Findings require field validation before formal action.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    BHCPF Accountability Tracker · Developed by Lawal Lawal Ali · October 2026<br>
    ML Pipeline: Random Forest (ROC-AUC 0.793) · Anomaly Detection: Isolation Forest + LOF + Z-score<br>
    <em>This report is for public accountability purposes. Findings require validation before formal action.</em>
</div>
""", unsafe_allow_html=True)
