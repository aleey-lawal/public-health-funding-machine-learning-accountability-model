# BHCPF Accountability Tracker

**ML-powered public accountability dashboard for the Basic Health Care Provision Fund (BHCPF)**

> Developed by **Lawal Lawal Ali** · Published **October 2026**

---

## Overview

This project tracks ₦57.1 billion disbursed across 37 Nigerian states and FCT under the BHCPF programme, Q1–Q3 2025. It combines unsupervised anomaly detection with a supervised facility-level risk model to flag states and primary health care (PHC) facilities warranting further investigation.

**Live dashboard:** [bhcpf-tracker.streamlit.app](https://bhcpf-tracker.streamlit.app) *(replace with your deployed URL)*  
**Static HTML version:** [GitHub Pages](https://your-username.github.io/bhcpf-accountability) *(replace with your URL)*

This project extends the [BHCPF Interactive Accountability Map](https://legendary-churros-fc5ebd.netlify.app) — a citizen-facing tool where anyone can click on a facility, navigate to its location, and verify whether it is functional, partially functional, or non-functional. This ML version adds anomaly detection and a facility risk model to flag states and PHCs warranting further investigation.

---

## Key Findings

| Tier | States | Description |
|------|--------|-------------|
| 🔴 CRITICAL | BORNO, KANO, KATSINA | Flagged by all 3 anomaly methods — extreme Q2 spikes |
| 🟡 CONCERN | FCT, SOKOTO, ZAMFARA | 2/3 methods — sustained Q2+Q3 elevation |
| 🔵 WATCH | LAGOS, YOBE | 1/3 methods — single-method outlier signal |
| 🟢 CLEAN | 29 states | No anomaly flags raised |

- **501 facilities** scored HIGH risk (≥ 0.80) across 8,309 tracked PHCs  
- **Top risk cluster:** Illela and Isa LGAs, Sokoto State (score: 0.9907)  
- **Model ROC-AUC:** 0.793 (Random Forest, 300 trees)

---

## Project Structure

```
bhcpf_project/
├── app.py                    # Streamlit dashboard (5 tabs, fully interactive)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore                # Files excluded from version control
├── bhcpf_dashboard.html      # Static HTML version (GitHub Pages)
│
├── notebooks/
│   ├── 01_data_extraction.ipynb     # Load & clean Q1–Q3 disbursement data
│   ├── 02_feature_engineering.ipynb # Build state-level feature matrix
│   ├── 03_anomaly_detection.ipynb   # Isolation Forest + LOF + Z-score
│   ├── 04_facility_risk_model.ipynb # Random Forest classifier
│   └── 05_reporting.ipynb           # Charts and summary statistics
│
├── data/
│   ├── raw/                  # Original Excel + PDF files (gitignored)
│   └── sample_data.csv       # Anonymised sample for demo purposes
│
├── output/
│   ├── state_features.csv    # 37 × 18 state feature matrix
│   ├── anomaly_results.csv   # Per-state anomaly scores and tiers
│   └── facility_risk_scores.csv  # 8,309 facilities with ML risk scores
│
└── .streamlit/
    └── config.toml           # Streamlit theme configuration
```

---

## Methodology

### 1. Data Sources
- **BHCPF disbursements:** Q1 & Q2 from Excel (NPHCDA); Q3 from PDF extraction
- **PHC facility registry:** 8,309 facilities, NPHCDA national registry
- **Population estimates:** NPC 2023 projections (⚠️ based on 2006 census growth rates)

### 2. Anomaly Detection (Unsupervised — State Level)
Three independent methods, no labelled ground truth required:

| Method | Algorithm | Threshold |
|--------|-----------|-----------|
| Isolation Forest | contamination = 0.15 | Score < 0 |
| Local Outlier Factor | n_neighbors = 5 | Score > 1.5 |
| Z-score composite | 9 features | \|z\| ≥ 2.0 on ≥ 3 features |

Consensus scoring (0–3 methods) classifies each state as CRITICAL / CONCERN / WATCH / CLEAN.

### 3. Facility Risk Model (Supervised — Facility Level)
- **Target:** AT RISK (PARTIAL + NON-FUNCTIONAL) vs FUNCTIONAL (L1 + L2)
- **Algorithm:** Random Forest (n_estimators=300, max_depth=10, class_weight='balanced')
- **Features (14):** LGA encoding, efficiency index, functionality score, consensus score, YTD per facility, % non-functional, YTD per million pop, QoQ volatility, gateway entropy
- **Evaluation:** Stratified 20% holdout · ROC-AUC **0.793** · Recall AT RISK **0.64**

## Disclaimer

This report is for public accountability purposes only. Anomaly flags do not constitute proof of misuse; they identify states and facilities warranting further investigation by the appropriate authorities. All findings require validation before formal action.

---

*Developed by Lawal Lawal Ali · October 2026 · PPDC Nigeria*
