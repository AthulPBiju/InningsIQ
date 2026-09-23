# =============================================================================
#  InningsIQ — Cricket Match Analytics & Prediction Engine
#  Author  : Athul P Biju
#  Version : 1.0.0
#  Dataset : IPL 2007–2026 Complete Ball-by-Ball Dataset (Kaggle)
#  Stack   : Python · Pandas · Scikit-learn · XGBoost · Streamlit
# =============================================================================
#
#  Single-file architecture:
#   1. Configuration & constants
#   2. Data loading & preprocessing
#   3. Feature engineering  (H2H, venue win-%, team form, toss, etc.)
#   4. Model training       (XGBoost classifier + regressor)
#   5. Streamlit UI         (dashboard tabs, metric cards, charts)
#
# =============================================================================

import os
import warnings
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  PAGE CONFIG  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="InningsIQ — Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 1.  GLOBAL CONSTANTS & CSS THEME
# ---------------------------------------------------------------------------
APP_VERSION = "1.0.0"
MODEL_CACHE_DIR = Path(".inningsiq_cache")
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Premium dark-themed sports-analytics CSS
CUSTOM_CSS = """
<style>
/* ── Base & fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background & surface ────────────────────────────────────── */
.stApp { background: #0d1117; }
section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

/* ── Hero banner ─────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #1a1f2e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero-title {
    font-size: 3rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(90deg, #58a6ff, #7ee787, #f78166);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle { color: #8b949e; font-size: 1.05rem; margin-top: 0.4rem; }

/* ── Metric card ─────────────────────────────────────────────── */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    height: 100%;
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #58a6ff; }
.metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }

/* ── Section header ──────────────────────────────────────────── */
.section-header {
    font-size: 1.15rem; font-weight: 600; color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding-left: 0.8rem; margin: 1.2rem 0 0.8rem;
}

/* ── Win probability bar ─────────────────────────────────────── */
.prob-bar-wrap { background: #21262d; border-radius: 999px; height: 22px; overflow: hidden; margin: 0.4rem 0; }
.prob-bar-fill { height: 100%; border-radius: 999px; display: flex; align-items: center; padding-left: 10px;
    font-size: 0.75rem; font-weight: 600; color: #fff; transition: width .4s ease; }

/* ── Alert / info boxes ──────────────────────────────────────── */
.alert-info { background:#1c2d3a; border-left:3px solid #58a6ff; border-radius:8px; padding:.8rem 1rem; color:#cdd9e5; font-size:.9rem; }
.alert-success { background:#1a2e1a; border-left:3px solid #7ee787; border-radius:8px; padding:.8rem 1rem; color:#c3e6cb; font-size:.9rem; }
.alert-warning { background:#2d2316; border-left:3px solid #d29922; border-radius:8px; padding:.8rem 1rem; color:#e6c88a; font-size:.9rem; }

/* ── Tabs ────────────────────────────────────────────────────── */
button[data-baseweb="tab"] { color: #8b949e !important; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }

/* ── Streamlit metric tweaks ─────────────────────────────────── */
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }
</style>
"""

# ---------------------------------------------------------------------------
# 2.  DATA LOADING  — cached so the CSV is read only once per session
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(matches_path: str, deliveries_path: str):
    """
    Load matches.csv and deliveries.csv from the provided paths.
    Returns cleaned DataFrames ready for feature engineering.
    """
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # ── Standardise column names (lower-snake) ────────────────────────────
    matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
    deliveries.columns = deliveries.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Essential column aliases (handles different Kaggle versions) ──────
    col_aliases = {
        "match_id": ["id", "match_id"],
        "season":   ["season"],
        "date":     ["date"],
        "team1":    ["team1"],
        "team2":    ["team2"],
        "venue":    ["venue"],
        "toss_winner":  ["toss_winner"],
        "toss_decision":["toss_decision"],
        "winner":   ["winner"],
        "result":   ["result"],
        "result_margin": ["result_margin"],
        "player_of_match": ["player_of_match"],
        "city":     ["city"],
    }
    for canonical, candidates in col_aliases.items():
        for c in candidates:
            if c in matches.columns and canonical not in matches.columns:
                matches.rename(columns={c: canonical}, inplace=True)

    # ── Date parsing ──────────────────────────────────────────────────────
    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce", dayfirst=True)
        matches.sort_values("date", inplace=True)
        matches.reset_index(drop=True, inplace=True)

    # ── Drop rows with no winner (ties, no-results) ───────────────────────
    matches = matches[matches["winner"].notna() & (matches["winner"] != "")].copy()

    # ── Normalize team names (strip whitespace) ───────────────────────────
    for col in ["team1", "team2", "winner", "toss_winner"]:
        if col in matches.columns:
            matches[col] = matches[col].str.strip()

    if "batting_team" in deliveries.columns:
        deliveries["batting_team"] = deliveries["batting_team"].str.strip()
    if "bowling_team" in deliveries.columns:
        deliveries["bowling_team"] = deliveries["bowling_team"].str.strip()

    return matches, deliveries


# ---------------------------------------------------------------------------
# 3.  FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def compute_team_stats(matches: pd.DataFrame):
    """
    Return a dict of per-team aggregated statistics:
    total matches, wins, win-rate, seasons active.
    """
    all_teams = pd.concat([matches["team1"], matches["team2"]]).unique()
    stats = {}
    for team in all_teams:
        played = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        won = matches[matches["winner"] == team]
        stats[team] = {
            "played": len(played),
            "wins": len(won),
            "win_rate": len(won) / len(played) if len(played) > 0 else 0,
        }
    return stats


def compute_h2h(matches: pd.DataFrame, team_a: str, team_b: str):
    """
    Head-to-head record between team_a and team_b.
    Returns dict with wins_a, wins_b, draws, total.
    """
    h2h = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
        ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ]
    wins_a = (h2h["winner"] == team_a).sum()
    wins_b = (h2h["winner"] == team_b).sum()
    return {
        "total": len(h2h),
        "wins_a": int(wins_a),
        "wins_b": int(wins_b),
        "win_pct_a": wins_a / len(h2h) * 100 if len(h2h) > 0 else 50.0,
        "win_pct_b": wins_b / len(h2h) * 100 if len(h2h) > 0 else 50.0,
    }


def compute_venue_stats(matches: pd.DataFrame, venue: str, team: str):
    """
    Win percentage of *team* at *venue*.
    """
    venue_matches = matches[matches["venue"] == venue]
    team_venue = venue_matches[
        (venue_matches["team1"] == team) | (venue_matches["team2"] == team)
    ]
    wins = (team_venue["winner"] == team).sum()
    total = len(team_venue)
    return wins / total * 100 if total > 0 else 50.0


def compute_recent_form(matches: pd.DataFrame, team: str, n: int = 5):
    """
    Win-rate of *team* over its last *n* completed matches.
    Returns a float in [0, 1].
    """
    team_matches = matches[
        (matches["team1"] == team) | (matches["team2"] == team)
    ].tail(n)
    if len(team_matches) == 0:
        return 0.5
    wins = (team_matches["winner"] == team).sum()
    return wins / len(team_matches)


def compute_toss_advantage(matches: pd.DataFrame, team: str, venue: str):
    """
    Proportion of times *team* won the match having won the toss at *venue*.
    """
    venue_m = matches[matches["venue"] == venue]
    toss_won = venue_m[venue_m["toss_winner"] == team]
    if len(toss_won) == 0:
        return 0.5
    match_won = (toss_won["winner"] == team).sum()
    return match_won / len(toss_won)


@st.cache_data(show_spinner=False)
def engineer_features(_matches: pd.DataFrame):
    """
    Build the training feature matrix for every historical match.

    Features per match
    ------------------
    h2h_win_pct_team1     : head-to-head win % for team1 vs team2
    venue_win_pct_team1   : team1 venue win %
    venue_win_pct_team2   : team2 venue win %
    form_team1            : last-5 win rate for team1
    form_team2            : last-5 win rate for team2
    toss_adv_team1        : toss→match win conv. for team1 at venue
    toss_adv_team2        : same for team2
    toss_winner_is_team1  : binary 1/0
    result_margin         : target for regression
    winner_label          : binary 1 (team1 wins) / 0 (team2 wins)
    """
    rows = []
    matches = _matches.reset_index(drop=True)

    for idx, row in matches.iterrows():
        # Only use historical data up to (but not including) the current match
        history = matches.iloc[:idx]
        if len(history) < 10:
            continue  # not enough history for meaningful features

        t1, t2, venue = row["team1"], row["team2"], row["venue"]

        h2h = compute_h2h(history, t1, t2)
        feat = {
            "h2h_win_pct_team1":    h2h["win_pct_a"],
            "venue_win_pct_team1":  compute_venue_stats(history, venue, t1),
            "venue_win_pct_team2":  compute_venue_stats(history, venue, t2),
            "form_team1":           compute_recent_form(history, t1, 5) * 100,
            "form_team2":           compute_recent_form(history, t2, 5) * 100,
            "toss_adv_team1":       compute_toss_advantage(history, t1, venue) * 100,
            "toss_adv_team2":       compute_toss_advantage(history, t2, venue) * 100,
            "toss_winner_is_team1": 1 if row.get("toss_winner") == t1 else 0,
            "h2h_total":            h2h["total"],
            "result_margin":        pd.to_numeric(row.get("result_margin", 0), errors="coerce") or 0,
            "winner_label":         1 if row["winner"] == t1 else 0,
            "team1":                t1,
            "team2":                t2,
            "venue":                venue,
            "season":               row.get("season", ""),
        }
        rows.append(feat)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 4.  MODEL TRAINING
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "h2h_win_pct_team1",
    "venue_win_pct_team1", "venue_win_pct_team2",
    "form_team1", "form_team2",
    "toss_adv_team1", "toss_adv_team2",
    "toss_winner_is_team1",
    "h2h_total",
]


def _cache_key(df: pd.DataFrame) -> str:
    """SHA-1 of the feature matrix shape + first/last row hash for quick cache validation."""
    sig = f"{df.shape}_{df.iloc[0].to_json()}_{df.iloc[-1].to_json()}"
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


@st.cache_resource(show_spinner=False)
def train_models(_feat_df: pd.DataFrame):
    """
    Train:
      • XGBoost Classifier  → predict match winner (team1 or team2)
      • XGBoost Regressor   → predict winning margin (runs/wickets)

    Returns (classifier, regressor, metrics_dict, feature_importance_df)
    """
    df = _feat_df.dropna(subset=FEATURE_COLS + ["winner_label", "result_margin"])
    X = df[FEATURE_COLS].values
    y_cls = df["winner_label"].values
    y_reg = df["result_margin"].values.astype(float)

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )

    # ── Classifier ────────────────────────────────────────────────────────
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_train, yc_train)
    yc_pred = clf.predict(X_test)
    acc = accuracy_score(yc_test, yc_pred)
    cls_report = classification_report(yc_test, yc_pred, output_dict=True)

    # ── Regressor ─────────────────────────────────────────────────────────
    reg = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    reg.fit(X_train, yr_train)
    yr_pred = reg.predict(X_test)
    mae = mean_absolute_error(yr_test, yr_pred)
    r2 = r2_score(yr_test, yr_pred)

    # ── Feature importance ────────────────────────────────────────────────
    fi_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    metrics = {
        "accuracy": acc,
        "cls_report": cls_report,
        "mae": mae,
        "r2": r2,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    return clf, reg, metrics, fi_df


# ---------------------------------------------------------------------------
# 5.  INFERENCE — real-time prediction for a user-selected fixture
# ---------------------------------------------------------------------------

def predict_match(
    clf, reg,
    matches: pd.DataFrame,
    team1: str, team2: str, venue: str,
    toss_winner: str,
):
    """
    Generate win-probability and predicted margin for the chosen fixture.
    Uses all available historical data as context.
    """
    h2h = compute_h2h(matches, team1, team2)
    feat_row = np.array([[
        h2h["win_pct_a"],
        compute_venue_stats(matches, venue, team1),
        compute_venue_stats(matches, venue, team2),
        compute_recent_form(matches, team1, 5) * 100,
        compute_recent_form(matches, team2, 5) * 100,
        compute_toss_advantage(matches, team1, venue) * 100,
        compute_toss_advantage(matches, team2, venue) * 100,
        1 if toss_winner == team1 else 0,
        h2h["total"],
    ]])

    proba = clf.predict_proba(feat_row)[0]   # [prob_team2_wins, prob_team1_wins]
    p_team1 = float(proba[1]) * 100
    p_team2 = 100.0 - p_team1
    margin   = max(0, float(reg.predict(feat_row)[0]))

    return {
        "p_team1": round(p_team1, 1),
        "p_team2": round(p_team2, 1),
        "predicted_margin": round(margin, 1),
        "h2h": h2h,
        "form_team1": round(compute_recent_form(matches, team1, 5) * 100, 1),
        "form_team2": round(compute_recent_form(matches, team2, 5) * 100, 1),
        "venue_pct_team1": round(compute_venue_stats(matches, venue, team1), 1),
        "venue_pct_team2": round(compute_venue_stats(matches, venue, team2), 1),
    }


# ---------------------------------------------------------------------------
# 6.  STREAMLIT UI HELPERS
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, sublabel: str = ""):
    """Render a styled KPI card."""
    sub = f'<div class="metric-label">{sublabel}</div>' if sublabel else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def prob_bar(team: str, pct: float, color: str = "#58a6ff"):
    """Render a horizontal probability bar."""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin:4px 0;">
            <span style="color:#8b949e; font-size:.85rem; width:220px; text-align:right;">{team}</span>
            <div class="prob-bar-wrap" style="flex:1;">
                <div class="prob-bar-fill" style="width:{pct}%; background:{color};">{pct:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 7.  CHART HELPERS  (Plotly dark theme)
# ---------------------------------------------------------------------------

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(family="Inter", color="#c9d1d9"),
)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = "#58a6ff"):
    fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[color])
    fig.update_layout(**PLOTLY_THEME, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#21262d")
    return fig


def gauge_chart(value: float, title: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": title, "font": {"size": 14, "color": "#8b949e"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#30363d"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#21262d"},
                {"range": [40, 60], "color": "#21262d"},
                {"range": [60, 100],"color": "#21262d"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": value},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="Inter"),
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# 8.  MAIN APPLICATION
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero banner ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🏏 InningsIQ</div>
            <div class="hero-subtitle">
                Cricket Match Analytics &amp; Prediction Engine &nbsp;·&nbsp; IPL Edition
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar — data upload ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Data Configuration")
        st.markdown("---")

        use_demo = st.checkbox("▶ Use demo / sample data", value=False,
            help="Enable to generate synthetic data for a quick walkthrough.")

        st.markdown("**Or upload your Kaggle CSVs:**")
        matches_file    = st.file_uploader("matches.csv",    type="csv", key="m")
        deliveries_file = st.file_uploader("deliveries.csv", type="csv", key="d")

        st.markdown("---")
        st.markdown(
            "<small style='color:#57606a;'>Dataset: "
            "<a href='https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020' "
            "target='_blank' style='color:#58a6ff;'>IPL 2007–2026 on Kaggle</a></small>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<small style='color:#57606a;'>InningsIQ v{APP_VERSION}</small>", unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────
    matches = deliveries = feat_df = clf = reg = metrics = fi_df = None

    if use_demo:
        with st.spinner("Generating synthetic IPL demo data…"):
            matches, deliveries = _generate_demo_data()
        st.markdown('<div class="alert-warning">⚠ Demo mode active — using synthetic data. Results are illustrative only.</div>', unsafe_allow_html=True)

    elif matches_file and deliveries_file:
        try:
            import tempfile, shutil
            with tempfile.TemporaryDirectory() as tmp:
                mp = os.path.join(tmp, "matches.csv")
                dp = os.path.join(tmp, "deliveries.csv")
                with open(mp, "wb") as f: f.write(matches_file.read())
                with open(dp, "wb") as f: f.write(deliveries_file.read())
                with st.spinner("Loading datasets…"):
                    matches, deliveries = load_data(mp, dp)
            st.markdown(f'<div class="alert-success">✅ Loaded {len(matches):,} matches and {len(deliveries):,} deliveries.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Data loading failed: {e}")
            return
    else:
        _show_landing_page()
        return

    # ── Feature engineering ───────────────────────────────────────────────
    with st.spinner("Engineering features from historical records…"):
        feat_df = engineer_features(matches)

    if len(feat_df) < 50:
        st.error("Not enough historical data to train a reliable model (need ≥ 50 matches with history).")
        return

    # ── Model training ────────────────────────────────────────────────────
    with st.spinner("Training XGBoost models — this runs once and is cached…"):
        clf, reg, metrics, fi_df = train_models(feat_df)

    # ── Tab layout ────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🎯 Match Predictor",
        "📊 Team Analytics",
        "🏟️ Venue Insights",
        "🤖 Model Performance",
        "📈 Season Trends",
    ])

    all_teams  = sorted(set(matches["team1"].tolist() + matches["team2"].tolist()))
    all_venues = sorted(matches["venue"].dropna().unique().tolist())

    # ================================================================
    # TAB 1 — MATCH PREDICTOR
    # ================================================================
    with tabs[0]:
        section_header("Configure Your Match")
        c1, c2, c3 = st.columns(3)
        with c1:
            team1 = st.selectbox("🔵 Team 1", all_teams, index=0, key="t1")
        with c2:
            team2_opts = [t for t in all_teams if t != team1]
            team2 = st.selectbox("🔴 Team 2", team2_opts, index=min(1, len(team2_opts)-1), key="t2")
        with c3:
            venue = st.selectbox("🏟️ Venue", all_venues, key="v")

        c4, c5 = st.columns(2)
        with c4:
            toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2], key="tw")
        with c5:
            toss_decision = st.selectbox("📋 Toss Decision", ["bat", "field"], key="td")

        predict_btn = st.button("🚀 Predict Match Outcome", use_container_width=True, type="primary")

        if predict_btn:
            with st.spinner("Running prediction engine…"):
                result = predict_match(clf, reg, matches, team1, team2, venue, toss_winner)

            st.markdown("---")
            section_header("Prediction Results")

            winner_name  = team1 if result["p_team1"] >= result["p_team2"] else team2
            winner_prob  = max(result["p_team1"], result["p_team2"])
            winner_color = "#7ee787" if result["p_team1"] >= result["p_team2"] else "#f78166"

            # Winner announcement
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#1a2e1a,#0d1117);
                            border:1px solid {winner_color};border-radius:14px;
                            padding:1.5rem;text-align:center;margin:.8rem 0;">
                    <div style="color:#8b949e;font-size:.85rem;text-transform:uppercase;letter-spacing:1px;">Predicted Winner</div>
                    <div style="font-size:2.4rem;font-weight:800;color:{winner_color};margin:.4rem 0;">{winner_name}</div>
                    <div style="color:#8b949e;font-size:.95rem;">Win Probability: <strong style="color:{winner_color};">{winner_prob:.1f}%</strong></div>
                    <div style="color:#8b949e;font-size:.9rem;margin-top:.3rem;">Estimated Margin: <strong style="color:#d29922;">{result['predicted_margin']:.0f} runs/wickets</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Probability gauges
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(gauge_chart(result["p_team1"], f"{team1} Win %", "#58a6ff"), use_container_width=True)
            with g2:
                st.plotly_chart(gauge_chart(result["p_team2"], f"{team2} Win %", "#f78166"), use_container_width=True)

            # Probability bars
            section_header("Win Probability Distribution")
            prob_bar(team1, result["p_team1"], "#58a6ff")
            prob_bar(team2, result["p_team2"], "#f78166")

            # Feature breakdown
            section_header("Feature Breakdown")
            fd1, fd2, fd3, fd4 = st.columns(4)
            with fd1: metric_card("H2H Matches",    str(result["h2h"]["total"]))
            with fd2: metric_card(f"{team1} Form",  f"{result['form_team1']:.0f}%", "Last 5 matches")
            with fd3: metric_card(f"{team2} Form",  f"{result['form_team2']:.0f}%", "Last 5 matches")
            with fd4: metric_card("Toss Winner",    toss_winner)

            # H2H comparison radar
            section_header("Head-to-Head Overview")
            h2h_fig = go.Figure(go.Bar(
                x=[team1, team2],
                y=[result["h2h"]["wins_a"], result["h2h"]["wins_b"]],
                marker_color=["#58a6ff", "#f78166"],
                text=[result["h2h"]["wins_a"], result["h2h"]["wins_b"]],
                textposition="outside",
            ))
            h2h_fig.update_layout(
                **PLOTLY_THEME,
                title=f"H2H Wins: {team1} vs {team2}",
                yaxis_title="Wins",
                showlegend=False,
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(h2h_fig, use_container_width=True)

    # ================================================================
    # TAB 2 — TEAM ANALYTICS
    # ================================================================
    with tabs[1]:
        section_header("All-Time Team Performance")
        team_stats = compute_team_stats(matches)
        stats_df = pd.DataFrame([
            {"Team": k, "Played": v["played"], "Wins": v["wins"],
             "Win Rate (%)": round(v["win_rate"] * 100, 1)}
            for k, v in team_stats.items()
        ]).sort_values("Win Rate (%)", ascending=False).reset_index(drop=True)

        # Leaderboard KPIs
        k1, k2, k3 = st.columns(3)
        with k1: metric_card("Total Teams", str(len(stats_df)))
        with k2: metric_card("Total Matches", f"{len(matches):,}")
        with k3: metric_card("Seasons", str(matches["season"].nunique()) if "season" in matches.columns else "—")

        st.markdown("---")

        # Win rate bar chart
        fig_wr = bar_chart(stats_df.head(12), x="Team", y="Win Rate (%)",
                           title="Top 12 Teams by Win Rate", color="#7ee787")
        st.plotly_chart(fig_wr, use_container_width=True)

        # Detailed table
        section_header("Full Leaderboard")
        st.dataframe(
            stats_df.style.background_gradient(subset=["Win Rate (%)"], cmap="Blues"),
            use_container_width=True, height=350,
        )

        # Deep-dive on selected team
        section_header("Team Deep-Dive")
        sel_team = st.selectbox("Select team", all_teams, key="sel_team_dd")
        if sel_team:
            team_matches = matches[
                (matches["team1"] == sel_team) | (matches["team2"] == sel_team)
            ].copy()
            if "season" in team_matches.columns:
                season_perf = team_matches.groupby("season").apply(
                    lambda g: (g["winner"] == sel_team).sum() / len(g) * 100
                ).reset_index(columns=["season", "win_pct"])  if False else (
                    team_matches.groupby("season").apply(
                        lambda g: pd.Series({
                            "win_pct": (g["winner"] == sel_team).sum() / len(g) * 100,
                            "matches": len(g)
                        })
                    ).reset_index()
                )
                fig_sp = px.line(
                    season_perf, x="season", y="win_pct",
                    markers=True, title=f"{sel_team} — Win % by Season",
                    color_discrete_sequence=["#58a6ff"],
                )
                fig_sp.update_layout(**PLOTLY_THEME, height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_sp, use_container_width=True)

    # ================================================================
    # TAB 3 — VENUE INSIGHTS
    # ================================================================
    with tabs[2]:
        section_header("Venue Statistics")
        venue_stats = matches.groupby("venue").agg(
            matches_hosted=("winner", "count")
        ).reset_index().sort_values("matches_hosted", ascending=False)

        k1, k2 = st.columns(2)
        with k1: metric_card("Total Venues", str(len(venue_stats)))
        with k2: metric_card("Most Used Venue", venue_stats.iloc[0]["venue"] if len(venue_stats) > 0 else "—",
                              str(venue_stats.iloc[0]["matches_hosted"]) + " matches")

        fig_v = bar_chart(
            venue_stats.head(15), x="venue", y="matches_hosted",
            title="Top 15 Venues by Matches Hosted", color="#d29922"
        )
        fig_v.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_v, use_container_width=True)

        section_header("Team Performance at Venue")
        sv_col1, sv_col2 = st.columns(2)
        with sv_col1:
            sel_venue_team = st.selectbox("Team", all_teams, key="vt")
        with sv_col2:
            sel_venue = st.selectbox("Venue", all_venues, key="vv")

        vp = compute_venue_stats(matches, sel_venue, sel_venue_team)
        vm = matches[(matches["venue"] == sel_venue) &
                     ((matches["team1"] == sel_venue_team) | (matches["team2"] == sel_venue_team))]
        v1, v2, v3 = st.columns(3)
        with v1: metric_card("Win % at Venue", f"{vp:.1f}%")
        with v2: metric_card("Matches Played", str(len(vm)))
        with v3: metric_card("Wins", str((vm["winner"] == sel_venue_team).sum()))

        # Toss decision distribution at venue
        if "toss_decision" in matches.columns:
            td_venue = matches[matches["venue"] == sel_venue]["toss_decision"].value_counts().reset_index()
            td_venue.columns = ["Decision", "Count"]
            if len(td_venue) > 0:
                fig_td = px.pie(td_venue, names="Decision", values="Count",
                                title=f"Toss Decisions at {sel_venue}",
                                color_discrete_sequence=["#58a6ff", "#f78166"])
                fig_td.update_layout(**PLOTLY_THEME, height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_td, use_container_width=True)

    # ================================================================
    # TAB 4 — MODEL PERFORMANCE
    # ================================================================
    with tabs[3]:
        section_header("Model Metrics Overview")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: metric_card("Classifier Accuracy", f"{metrics['accuracy']*100:.1f}%", "XGBoost")
        with mc2: metric_card("Training Samples",    f"{metrics['train_size']:,}")
        with mc3: metric_card("Test Samples",        f"{metrics['test_size']:,}")
        with mc4: metric_card("Margin MAE",          f"{metrics['mae']:.1f}", "runs/wickets")

        section_header("Feature Importance")
        fi_fig = bar_chart(fi_df, x="feature", y="importance",
                           title="XGBoost Feature Importances (Classifier)", color="#7ee787")
        fi_fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fi_fig, use_container_width=True)

        section_header("Classification Report")
        report = metrics["cls_report"]
        rdf = pd.DataFrame({
            "Class": ["Team 2 Wins (0)", "Team 1 Wins (1)"],
            "Precision": [report["0"]["precision"], report["1"]["precision"]],
            "Recall":    [report["0"]["recall"],    report["1"]["recall"]],
            "F1-Score":  [report["0"]["f1-score"],  report["1"]["f1-score"]],
            "Support":   [report["0"]["support"],   report["1"]["support"]],
        })
        st.dataframe(rdf.style.format({
            "Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}"
        }), use_container_width=True)

        section_header("Regression Model (Margin Predictor)")
        st.markdown(
            f"""
            <div class="alert-info">
            📐 <strong>Mean Absolute Error:</strong> {metrics['mae']:.2f} runs/wickets &nbsp;|&nbsp;
            📈 <strong>R² Score:</strong> {metrics['r2']:.4f}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ================================================================
    # TAB 5 — SEASON TRENDS
    # ================================================================
    with tabs[4]:
        section_header("IPL Season Overview")
        if "season" in matches.columns:
            season_agg = matches.groupby("season").agg(
                total_matches=("winner", "count"),
                unique_teams=("team1", lambda x: pd.Series(
                    list(x) + list(matches.loc[x.index, "team2"])
                ).nunique()),
            ).reset_index()

            fig_sm = px.bar(
                season_agg, x="season", y="total_matches",
                title="Matches per Season",
                color_discrete_sequence=["#58a6ff"],
            )
            fig_sm.update_layout(**PLOTLY_THEME, height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_sm, use_container_width=True)

            # Toss win → match win rate by season
            if "toss_winner" in matches.columns:
                matches["toss_match_win"] = matches["toss_winner"] == matches["winner"]
                toss_trend = matches.groupby("season")["toss_match_win"].mean().reset_index()
                toss_trend.columns = ["season", "toss_win_rate"]
                toss_trend["toss_win_rate"] *= 100

                fig_tt = px.line(
                    toss_trend, x="season", y="toss_win_rate",
                    markers=True, title="Toss-to-Win Conversion Rate by Season (%)",
                    color_discrete_sequence=["#d29922"],
                )
                fig_tt.update_layout(**PLOTLY_THEME, height=280, margin=dict(l=20, r=20, t=40, b=20))
                fig_tt.add_hline(y=50, line_dash="dash", line_color="#8b949e", annotation_text="50% baseline")
                st.plotly_chart(fig_tt, use_container_width=True)

        else:
            st.info("Season column not found in dataset.")

        # Result type distribution
        if "result" in matches.columns:
            section_header("Result Type Distribution")
            rt = matches["result"].value_counts().reset_index()
            rt.columns = ["Result Type", "Count"]
            fig_rt = px.pie(rt, names="Result Type", values="Count",
                            title="Match Result Types",
                            color_discrete_sequence=["#58a6ff", "#7ee787", "#f78166", "#d29922"])
            fig_rt.update_layout(**PLOTLY_THEME, height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_rt, use_container_width=True)


# ---------------------------------------------------------------------------
# 9.  LANDING PAGE (shown before data is loaded)
# ---------------------------------------------------------------------------

def _show_landing_page():
    st.markdown(
        """
        <div style="background:#161b22;border:1px solid #30363d;border-radius:14px;
                    padding:2rem;text-align:center;margin-top:1rem;">
            <div style="font-size:3rem;margin-bottom:.5rem;">🏏</div>
            <div style="font-size:1.4rem;font-weight:700;color:#e6edf3;">Welcome to InningsIQ</div>
            <div style="color:#8b949e;margin-top:.5rem;max-width:520px;margin-left:auto;margin-right:auto;font-size:.95rem;">
                Upload your IPL dataset CSVs using the sidebar, or enable
                <strong style="color:#58a6ff;">Demo Mode</strong> to explore the
                analytics dashboard with synthetic data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">🎯</div>'
            '<div class="metric-label" style="font-size:.95rem;color:#c9d1d9;margin-top:.5rem;">'
            'Match Winner Prediction<br><small style="color:#57606a;">XGBoost classifier with win probability %</small>'
            '</div></div>', unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">📊</div>'
            '<div class="metric-label" style="font-size:.95rem;color:#c9d1d9;margin-top:.5rem;">'
            'Feature Engineering<br><small style="color:#57606a;">H2H · Venue Win % · Team Form · Toss Advantage</small>'
            '</div></div>', unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">📈</div>'
            '<div class="metric-label" style="font-size:.95rem;color:#c9d1d9;margin-top:.5rem;">'
            'Season Trends & Analytics<br><small style="color:#57606a;">Venue stats · Season trends · Team leaderboards</small>'
            '</div></div>', unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# 10. DEMO DATA GENERATOR  (synthetic but realistic IPL-like data)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _generate_demo_data():
    """
    Generate a synthetic IPL-like dataset for demonstration purposes.
    Produces ~800 matches across 16 seasons (2008–2023).
    """
    rng = np.random.default_rng(42)
    teams = [
        "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
        "Kolkata Knight Riders", "Sunrisers Hyderabad", "Delhi Capitals",
        "Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants",
        "Gujarat Titans",
    ]
    venues = [
        "Wankhede Stadium", "MA Chidambaram Stadium", "Eden Gardens",
        "Narendra Modi Stadium", "Rajiv Gandhi International Cricket Stadium",
        "Feroz Shah Kotla", "Sawai Mansingh Stadium", "Punjab Cricket Association Stadium",
        "Brabourne Stadium", "DY Patil Stadium",
    ]
    seasons = list(range(2008, 2024))
    matches_per_season = 56

    rows = []
    match_id = 1
    for season in seasons:
        for _ in range(matches_per_season):
            t1, t2 = rng.choice(teams, size=2, replace=False)
            venue   = rng.choice(venues)
            toss_w  = rng.choice([t1, t2])
            toss_d  = rng.choice(["bat", "field"])
            # Simple win probability weighted by team index (lower index = stronger team)
            t1_idx, t2_idx = teams.index(t1), teams.index(t2)
            p_t1 = 0.5 + (t2_idx - t1_idx) * 0.03
            p_t1 = np.clip(p_t1, 0.2, 0.8)
            winner = t1 if rng.random() < p_t1 else t2
            margin = int(rng.integers(1, 120))
            result = rng.choice(["runs", "wickets"], p=[0.55, 0.45])
            date   = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=int(rng.integers(0, 55)))
            rows.append({
                "match_id": match_id, "season": season, "date": date,
                "team1": t1, "team2": t2, "venue": venue,
                "toss_winner": toss_w, "toss_decision": toss_d,
                "winner": winner, "result": result,
                "result_margin": margin, "player_of_match": "Player X",
                "city": venue.split()[0],
            })
            match_id += 1

    matches_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Minimal deliveries stub (just enough for the pipeline not to break)
    del_rows = []
    for _, m in matches_df.head(100).iterrows():
        for over in range(1, 21):
            for ball in range(1, 7):
                del_rows.append({
                    "match_id": m["match_id"],
                    "inning": 1,
                    "batting_team": m["team1"],
                    "bowling_team": m["team2"],
                    "over": over, "ball": ball,
                    "batsman_runs": int(rng.integers(0, 7)),
                    "extra_runs": 0,
                    "total_runs": int(rng.integers(0, 7)),
                    "is_wicket": int(rng.random() < 0.05),
                })
    deliveries_df = pd.DataFrame(del_rows)
    return matches_df, deliveries_df


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
