<div align="center">

# 🏏 InningsIQ
### Cricket Match Analytics & Prediction Engine — IPL Edition

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006AFF?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A market-ready, commercial-grade cricket analytics platform that predicts match winners, win probabilities, and winning margins using machine learning — powered by 18 seasons of IPL data.*

</div>

---

## 📌 Overview

**InningsIQ** transforms raw ball-by-ball cricket data into actionable pre-match intelligence. It engineers rich historical features — head-to-head records, venue win percentages, and rolling team form — feeds them into a calibrated **XGBoost** pipeline, and presents everything inside a **premium dark-themed Streamlit dashboard** that looks and feels like a professional sports analytics product.

Whether you are a data science student, an analytics professional, or a cricket enthusiast who wants to move beyond eye-test opinions, InningsIQ gives you a rigorous, reproducible, data-driven view of every IPL fixture.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **Match Winner Prediction** | XGBoost classifier outputs win probability % for both teams |
| 📐 **Margin Estimation** | XGBoost regressor predicts the expected winning margin in runs/wickets |
| 🔄 **Head-to-Head Engine** | Computes full H2H record and win percentage between any two teams |
| 🏟️ **Venue Win Rates** | Each team's historical win percentage broken down per stadium |
| 📈 **Rolling Team Form** | Last-5-match win rates computed at prediction time to capture momentum |
| 🪙 **Toss Advantage** | Venue-specific toss-to-win conversion rates per team |
| 📊 **Team Leaderboards** | All-time performance rankings with season-by-season drill-down |
| 📉 **Season Trends** | Toss-win correlation, result types, and matches-per-season visualisations |
| 🤖 **Model Report** | Live accuracy, MAE, F1-score, and feature importance charts |
| 🎮 **Demo Mode** | Fully synthetic IPL-like dataset — zero setup required |

---

## 📂 Dataset

> **IPL 2007 to 2026 Complete Ball-by-Ball Dataset**
> 🔗 https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

The project uses two CSV files from this dataset:

| File | Description |
|---|---|
| `matches.csv` | One row per match — teams, venue, toss, result, winner, margin |
| `deliveries.csv` | One row per ball — batting/bowling team, over, runs, wickets |

Download both files and place them anywhere accessible; you will upload them via the app's sidebar.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Wrangling | Pandas 2.x · NumPy 1.24+ |
| Machine Learning | XGBoost 2.x · Scikit-learn 1.3+ |
| Web Interface | Streamlit 1.32+ |
| Visualisation | Plotly 5.x |
| Styling | Custom CSS (dark sports-analytics theme) |

---

## 🚀 Setup & Run Instructions

### Prerequisites
- Python **3.10 or higher**
- pip (comes with Python)
- *(Optional)* A virtual environment tool such as `venv` or `conda`

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/AthulPBiju/InningsIQ.git
cd InningsIQ
```

---

### Step 2 — Create and activate a virtual environment *(recommended)*

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** For GPU-accelerated XGBoost training on large datasets, install the CUDA variant:
> `pip install xgboost[cuda]`

---

### Step 4 — Download the dataset

1. Visit the Kaggle dataset page linked above.
2. Click **Download** (you may need a free Kaggle account).
3. Extract the archive; locate `matches.csv` and `deliveries.csv`.
4. Keep them anywhere on your machine — you will upload them inside the app.

---

### Step 5 — Launch the app

```bash
streamlit run AthulPBiju_InningsIQ.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in any browser.

---

### Step 6 — Load data & explore

1. In the **sidebar**, upload `matches.csv` and `deliveries.csv`  
   — *or* tick **"Use demo / sample data"* for an instant walkthrough*.
2. The feature engineering and model training pipeline runs automatically (cached after the first run).
3. Navigate the five dashboard tabs:
   - **Match Predictor** → select two teams, a venue, and toss winner → click **Predict**.
   - **Team Analytics** → leaderboards and season-by-season win-rate trends.
   - **Venue Insights** → per-stadium statistics and toss-decision distributions.
   - **Model Performance** → classifier accuracy, regression MAE, feature importances.
   - **Season Trends** → matches per year, toss-win correlation over time.

---

## 📁 Project Structure

```
InningsIQ/
├── AthulPBiju_InningsIQ.py   # Single-file application (all logic + UI)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── AthulPBiju_ProjectReport.docx  # Formal academic project report
```

---

## 🎓 Architecture at a Glance

```
CSV Upload (matches.csv + deliveries.csv)
        │
        ▼
  Data Loader & Cleaner
  (column normalisation, date parsing, deduplication)
        │
        ▼
  Feature Engineering (per match, using prior history only)
  ├── Head-to-Head win %
  ├── Venue win % (team1 & team2)
  ├── Rolling 5-match form (team1 & team2)
  ├── Toss-advantage at venue (team1 & team2)
  └── H2H total count
        │
        ▼
  ┌─────────────────────────────┐
  │  XGBoost Classifier         │ ──► Win Probability (%)
  │  (winner prediction)        │
  └─────────────────────────────┘
  ┌─────────────────────────────┐
  │  XGBoost Regressor          │ ──► Predicted Margin (runs/wickets)
  │  (margin estimation)        │
  └─────────────────────────────┘
        │
        ▼
  Streamlit Dashboard
  (5 tabs · metric cards · Plotly charts · styled probability bars)
```

---

## ⚠️ Limitations & Disclaimer

- Predictions are based solely on historical aggregates and do not account for player availability, injuries, pitch conditions, or weather.
- This project is intended for educational and analytical purposes only.
- Do not use predictions for gambling or financial decisions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by <strong>Athul P Biju</strong> · Powered by IPL data · InningsIQ v1.0.0</sub>
</div>
