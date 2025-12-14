import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import sqlite3
import pickle
import time
from datetime import datetime
import os
from pathlib import Path

# Project paths (resilient: check data/ or fallback to repo root)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

def data_file(name):
    p = DATA_DIR / name
    if p.exists():
        return str(p)
    fallback = ROOT / name
    return str(fallback)

# File paths
DATA_FILE = data_file("ml_ready5.csv")
RAW_DATA_FILE = data_file("n.csv")
POINTS_DB = data_file("points1.db")
DB_FILE_NEW = data_file("super_predictions_v2.db")
CACHE_FILE = str(MODELS_DIR / "super_stats_cache_v2.pkl")
MODEL_XGB_FILE = str(MODELS_DIR / "super_tennis_xgb_v2.json")

# Initialize database
def initialize_db():
    conn = sqlite3.connect(DB_FILE_NEW)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player1 TEXT,
            player2 TEXT,
            match_date TEXT,
            predicted_winner TEXT,
            confidence REAL,
            predicted_sets TEXT,
            predicted_scores TEXT,
            actual_winner TEXT,
            timestamp TEXT,
            p1_serve_win REAL,
            p1_return_win REAL,
            p2_serve_win REAL,
            p2_return_win REAL,
            p1_momentum REAL,
            p2_momentum REAL,
            p1_fatigue REAL,
            p2_fatigue REAL,
            p1_clutch REAL,
            p2_clutch REAL,
            p1_consistency REAL,
            p2_consistency REAL,
            h2h_weight REAL,
            observation TEXT,
            is_correct INTEGER DEFAULT NULL  -- 1 if correct, 0 if incorrect, NULL if unknown
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized.")
# Compute detailed stats
def load_or_compute_stats(df, raw_df, n_matches=15, recent_n=5, long_term_years=2):
    start = time.time()
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    
    df["Event ID"] = df["Event ID"].astype(str)
    raw_df["Event ID"] = raw_df["Event ID"].astype(str)
    df = df.merge(raw_df[["Event ID", "Player 1", "Player 2"]], on="Event ID", how="left").sort_values(["Year", "Month", "Day"])
    singles_mask = (~df["Player 1"].str.contains("/").fillna(False)) & (~df["Player 2"].str.contains("/").fillna(False))
    df_singles = df[singles_mask].copy()
    
    print("Skipping points1.db load - clutch stats defaulting to average tiebreak wins.")
    players = set(df_singles["Player 1 ID"]).union(df_singles["Player 2 ID"])
    print(f"Computing stats for {len(players)} players...")
    player_stats = {}
    
    current_year = datetime.now().year
    long_term_cutoff = current_year - long_term_years
    
    for pid in players:
        matches = df_singles[(df_singles["Player 1 ID"] == pid) | (df_singles["Player 2 ID"] == pid)].tail(n_matches)
        if matches.empty:
            player_stats[pid] = {
                "Serve Win %": 50, "Return Win %": 50, "Recent Form": 50, "Momentum": 0, "Fatigue": 0,
                "Clutch %": 50, "Consistency": 50, "Break Point Edge": 50, "Max Streak": 0, "Ace Rate": 0, "Double Fault Rate": 0
            }
            continue
        
        stats = {
            "Long Serve Win %": 50, "Long Return Win %": 50,
            "Mid Serve Win %": 50, "Mid Return Win %": 50,
            "Recent Serve Win %": 50, "Recent Return Win %": 50
        }
        long_term = matches[matches["Year"] < long_term_cutoff]
        mid_term = matches[matches["Year"] >= long_term_cutoff]
        recent = matches.tail(recent_n)
        
        for period, period_name in [(long_term, "Long"), (mid_term, "Mid"), (recent, "Recent")]:
            if not period.empty:
                p1_serve = period[period["Player 1 ID"] == pid]["Player 1 First serve Win %"].mean()
                p2_serve = period[period["Player 2 ID"] == pid]["Player 2 First serve Win %"].mean()
                p1_return = period[period["Player 1 ID"] == pid]["Player 1 First serve return Win %"].mean()
                p2_return = period[period["Player 2 ID"] == pid]["Player 2 First serve return Win %"].mean()
                stats[f"{period_name} Serve Win %"] = np.nanmean([p1_serve, p2_serve]) if not np.isnan([p1_serve, p2_serve]).all() else 50
                stats[f"{period_name} Return Win %"] = np.nanmean([p1_return, p2_return]) if not np.isnan([p1_return, p2_return]).all() else 50
        
        stats["Serve Win %"] = stats["Recent Serve Win %"] * 0.6 + stats["Mid Serve Win %"] * 0.25 + stats["Long Serve Win %"] * 0.15
        stats["Return Win %"] = stats["Recent Return Win %"] * 0.6 + stats["Mid Return Win %"] * 0.25 + stats["Long Return Win %"] * 0.15
        
        wins = recent["Winner"].eq(1 if pid in recent["Player 1 ID"].values else 2).values
        weights = np.linspace(1, 2, len(wins))
        stats["Momentum"] = np.average(wins, weights=weights) * 100 if wins.size > 0 else 0
        stats["Recent Form"] = recent["Winner"].eq(1 if pid in recent["Player 1 ID"].values else 2).mean() * 100 if not recent.empty else 50
        
        durations = matches["Match Duration"].fillna(0)
        decay = np.exp(-np.arange(len(durations)) / 5)
        stats["Fatigue"] = np.average(durations, weights=decay) / 60 if durations.sum() > 0 else 0
        
        if pid in matches["Player 1 ID"].values:
            clutch = matches["Player 1 Tiebreak Wins"].mean() + matches["Player 1 Break points saved Made"].sum() / matches["Player 1 Break points saved Total"].sum() if matches["Player 1 Break points saved Total"].sum() > 0 else 0
            stats["Max Streak"] = matches["Player 1 Max games in a row"].mean()
            bp_edge = matches["Player 1 Break points converted"].sum() / matches["Player 1 Return games played"].sum() if matches["Player 1 Return games played"].sum() > 0 else 0
            ace_rate = matches["Player 1 Aces"].sum() / matches["Player 1 First serve Total"].sum() if matches["Player 1 First serve Total"].sum() > 0 else 0
            df_rate = matches["Player 1 Double faults"].sum() / matches["Player 1 First serve Total"].sum() if matches["Player 1 First serve Total"].sum() > 0 else 0
        else:
            clutch = matches["Player 2 Tiebreak Wins"].mean() + matches["Player 2 Break points saved Made"].sum() / matches["Player 2 Break points saved Total"].sum() if matches["Player 2 Break points saved Total"].sum() > 0 else 0
            stats["Max Streak"] = matches["Player 2 Max games in a row"].mean()
            bp_edge = matches["Player 2 Break points converted"].sum() / matches["Player 2 Return games played"].sum() if matches["Player 2 Return games played"].sum() > 0 else 0
            ace_rate = matches["Player 2 Aces"].sum() / matches["Player 2 First serve Total"].sum() if matches["Player 2 First serve Total"].sum() > 0 else 0
            df_rate = matches["Player 2 Double faults"].sum() / matches["Player 2 First serve Total"].sum() if matches["Player 2 First serve Total"].sum() > 0 else 0
        stats["Clutch %"] = clutch * 50 if not pd.isna(clutch) else 50
        stats["Break Point Edge"] = bp_edge * 100 if bp_edge > 0 else 50
        stats["Max Streak"] = 0 if pd.isna(stats["Max Streak"]) else stats["Max Streak"]
        stats["Ace Rate"] = ace_rate * 100
        stats["Double Fault Rate"] = df_rate * 100
        
        game_cols = [col for col in matches.columns if "Game" in col]
        if pid in matches["Player 1 ID"].values:
            game_wins = matches[game_cols].apply(lambda x: x.str.split("-")[0] if isinstance(x, str) else "0").astype(float)
        else:
            game_wins = matches[game_cols].apply(lambda x: x.str.split("-")[1] if isinstance(x, str) else "0").astype(float)
        stats["Consistency"] = 100 - np.std(game_wins.values.flatten()) * 10 if game_wins.size > 0 else 50
        
        player_stats[pid] = stats
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(player_stats, f)
    print(f"Computed stats in {time.time() - start:.2f}s")
    return player_stats
