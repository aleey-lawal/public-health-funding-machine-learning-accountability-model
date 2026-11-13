# BHCPF Accountability Tracker

**ML-powered public accountability dashboard for the Basic Health Care Provision Fund (BHCPF)**

> Developed by **Lawal Lawal Ali** · Published **October 2026**

---

## Overview

This project tracks ₦57.1 billion disbursed across 37 Nigerian states and FCT under the BHCPF programme, Q1–Q3 2025. It combines unsupervised anomaly detection with a supervised facility-level risk model to flag states and primary health care (PHC) facilities warranting further investigation.

**Live dashboard:** [bhcpf-tracker.streamlit.app](https://bhcpf-tracker.streamlit.app) *(replace with your deployed URL)*  
**Static HTML version:** [GitHub Pages](https://your-username.github.io/bhcpf-accountability) *(replace with your URL)*

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

### 4. Why No Confusion Matrix for Anomaly Detection?
Unsupervised anomaly detection produces a *score*, not a prediction against labelled truth. Without ground-truth labels at the state level (i.e. we don't know which states *actually* misused funds), precision/recall/F1/confusion matrices are not applicable. The confusion matrix appears only in the **Facility Risk Model** tab, where labelled facility statuses (L1/L2/Partial/Non-functional) provide a verifiable ground truth.

---

## Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/bhcpf-accountability.git
cd bhcpf-accountability

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`.

---

## Deploying to Streamlit Cloud

1. Push this repository to GitHub (see **GitHub Setup** below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file** to `app.py`
4. Click **Deploy** — your app gets a public URL in ~2 minutes

---

## GitHub Setup

```bash
# From inside the project folder:
git init
git add .
git commit -m "Initial commit: BHCPF Accountability Tracker

ML-powered dashboard tracking ₦57.1B in BHCPF disbursements
across 37 states + FCT, Q1–Q3 2025. Includes anomaly detection
(Isolation Forest + LOF + Z-score) and facility-level risk
scoring (Random Forest, ROC-AUC 0.793).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git remote add origin https://github.com/YOUR_USERNAME/bhcpf-accountability.git
git branch -M main
git push -u origin main
```

### GitHub Pages (Static HTML dashboard)
```bash
# Copy the HTML dashboard to a docs/ folder for GitHub Pages
mkdir docs
cp bhcpf_dashboard.html docs/index.html

git add docs/
git commit -m "Add GitHub Pages static dashboard"
git push

# Then in GitHub: Settings → Pages → Source: main branch /docs folder
```

---

## Disclaimer

This report is for public accountability purposes only. Anomaly flags do not constitute proof of misuse; they identify states and facilities warranting further investigation by the appropriate authorities. All findings require validation before formal action.

---

*Developed by Lawal Lawal Ali · October 2026 · PPDC Nigeria*
