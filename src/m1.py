import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import pickle
import time
from datetime import datetime
from collections import defaultdict
import os
from pathlib import Path

# Project paths
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
DB_FILE_NEW = data_file("predictions2.db")
CACHE_FILE = str(MODELS_DIR / "player_stats_cache.pkl")
MODEL_XGB_FILE = str(MODELS_DIR / "xgb_model.json")
MODEL_RF_FILE = str(MODELS_DIR / "rf_model.pkl")

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
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# Load or compute advanced player stats
def load_or_compute_stats(df, raw_df, n_matches=15):
    start = time.time()
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            player_stats = pickle.load(f)
        print(f"Loaded stats in {time.time() - start:.2f}s")
    else:
        player_stats = compute_player_stats(df, raw_df, n_matches)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(player_stats, f)
        print(f"Computed stats in {time.time() - start:.2f}s")
    return player_stats

def compute_player_stats(df, raw_df, n_matches):
    df = df.sort_values(["Year", "Month", "Day"]).copy()
    raw_df["Surface"] = raw_df.get("Surface", "Unknown")  # Assuming surface might be in raw data
    player_stats = defaultdict(dict)
    conn = sqlite3.connect(POINTS_DB)
    cursor = conn.cursor()
    
    for pid in set(df["Player 1 ID"]).union(df["Player 2 ID"]):
        matches = df[(df["Player 1 ID"] == pid) | (df["Player 2 ID"] == pid)].tail(n_matches)
        if matches.empty:
            continue
        weights = np.array([0.9 ** i for i in range(len(matches)-1, -1, -1)])
        
        # Basic stats
        for col in ["First serve Win %", "Second serve Win %", "First serve return Win %", "Second serve return Win %"]:
            p1_col = f"Player 1 {col}"
            p2_col = f"Player 2 {col}"
            if p1_col not in matches.columns or p2_col not in matches.columns:
                print(f"Warning: Missing column(s) {p1_col} or {p2_col} in data.")
                player_stats[pid][col] = 0
                continue
            p1_vals = matches[matches["Player 1 ID"] == pid][p1_col]
            p2_vals = matches[matches["Player 2 ID"] == pid][p2_col]
            vals = pd.concat([p1_vals, p2_vals])
            player_stats[pid][col] = np.average(vals, weights=weights[:len(vals)]) if not vals.empty else 0
        
        player_stats[pid]["Fatigue"] = matches["Match Duration"].mean() / 60 if "Match Duration" in matches.columns else 0
        player_stats[pid]["Tiebreak Wins"] = matches["Player 1 Tiebreak Wins"].sum() if pid in matches["Player 1 ID"].values else matches["Player 2 Tiebreak Wins"].sum()
        
        # Surface-specific stats (if available)
        event_ids = matches["Event ID"].tolist()
        surfaces = raw_df[raw_df["Event ID"].isin(event_ids)]["Surface"].value_counts().index[0] if "Surface" in raw_df.columns else "Unknown"
        player_stats[pid]["Surface Strength"] = matches["Winner"].map({1: 1, 2: 0}).mean() if pid in matches["Player 1 ID"].values else 1 - matches["Winner"].map({1: 1, 2: 0}).mean()
        
        # Point-level stats
        if event_ids:
            cursor.execute(f"SELECT \"Server\", \"Winner\", \"Point Type\", \"Game\" FROM points WHERE \"Event ID\" IN ({','.join('?' * len(event_ids))})", event_ids)
            points = pd.DataFrame(cursor.fetchall(), columns=["Server", "Winner", "Point Type", "Game"])
            clutch_points = points[points["Point Type"].isin(["BP", "SP"]) ]
            player_stats[pid]["Clutch %"] = len(clutch_points[clutch_points["Winner"] == pid]) / len(clutch_points) * 100 if len(clutch_points) > 0 else 0
            recent_points = points.tail(50)
            player_stats[pid]["Momentum"] = len(recent_points[recent_points["Winner"] == pid]) / 50 if len(recent_points) > 0 else 0
            break_points = points[points["Point Type"] == "BP"]
            player_stats[pid]["Break Conversion %"] = len(break_points[break_points["Winner"] == pid]) / len(break_points) * 100 if len(break_points) > 0 else 0
    
    conn.close()
    return player_stats

# Add features
def add_features(df, player_stats):
    start = time.time()
    feature_cols = ["First serve Win %", "Second serve Win %", "First serve return Win %", 
                    "Second serve return Win %", "Fatigue", "Clutch %", "Momentum", 
                    "Tiebreak Wins", "Surface Strength", "Break Conversion %"]
    
    for col in feature_cols:
        df[f"P1 {col}"] = df["Player 1 ID"].map(lambda x: player_stats.get(x, {}).get(col, 0))
        df[f"P2 {col}"] = df["Player 2 ID"].map(lambda x: player_stats.get(x, {}).get(col, 0))
    
    df["P1 H2H Strength"] = 0.5
    for idx, row in df.iterrows():
        p1, p2 = row["Player 1 ID"], row["Player 2 ID"]
        h2h = df[((df["Player 1 ID"] == p1) & (df["Player 2 ID"] == p2)) | 
                 ((df["Player 1 ID"] == p2) & (df["Player 2 ID"] == p1))].iloc[:-1]
        if not h2h.empty:
            weights = np.array([0.9 ** i for i in range(len(h2h)-1, -1, -1)])
            p1_wins = h2h.apply(lambda x: 1 if (x["Player 1 ID"] == p1 and x["Winner"] == 1) else 0, axis=1)
            df.at[idx, "P1 H2H Strength"] = np.average(p1_wins, weights=weights) if weights.sum() > 0 else 0.5
    
    print(f"Features added in {time.time() - start:.2f}s")
    return df

# Train ensemble model
def train_model(df):
    start = time.time()
    features = ["P1 First serve Win %", "P2 First serve Win %", "P1 Second serve Win %", "P2 Second serve Win %",
                "P1 First serve return Win %", "P2 First serve return Win %", "P1 Second serve return Win %", 
                "P2 Second serve return Win %", "P1 Fatigue", "P2 Fatigue", "P1 Clutch %", "P2 Clutch %",
                "P1 Momentum", "P2 Momentum", "P1 Tiebreak Wins", "P2 Tiebreak Wins", 
                "P1 Surface Strength", "P2 Surface Strength", "P1 Break Conversion %", "P2 Break Conversion %",
                "P1 H2H Strength"]
    X = df[features]
    y = df["Winner"].map({1: 1, 2: 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost
    xgb = XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.8, random_state=42)
    xgb.fit(X_train, y_train)
    xgb.save_model(MODEL_XGB_FILE)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    with open(MODEL_RF_FILE, 'wb') as f:
        pickle.dump(rf, f)
    
    # Ensemble prediction
    xgb_pred = xgb.predict(X_test)
    rf_pred = rf.predict(X_test)
    ensemble_pred = (xgb_pred + rf_pred) >= 1  # Majority vote
    print(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
    print(f"Training took {time.time() - start:.2f}s")
    return xgb, rf

# Dynamic scenario simulator
def generate_dynamic_scenario(p1_stats, p2_stats):
    p1_serve = p1_stats["First serve Win %"] / 100
    p2_serve = p2_stats["First serve Win %"] / 100
    p1_return = p1_stats["First serve return Win %"] / 100
    p2_return = p2_stats["First serve return Win %"] / 100
    p1_break = p1_stats["Break Conversion %"] / 100
    p2_break = p2_stats["Break Conversion %"] / 100
    p1_clutch = p1_stats["Clutch %"] / 100
    
    def simulate_set():
        p1_games, p2_games = 0, 0
        for _ in range(12):
            if np.random.random() < (p1_serve - p2_return):
                p1_games += 1
            else:
                p2_games += 1 if np.random.random() < p2_break else p2_games
            if p1_games >= 6 and p1_games - p2_games >= 2:
                return f"{p1_games}-{p2_games}", True
            if p2_games >= 6 and p2_games - p1_games >= 2:
                return f"{p1_games}-{p2_games}", False
        return "7-6" if np.random.random() < p1_clutch else "6-7", p1_clutch > 0.5
    
    sets, p1_sets = [], 0
    for _ in range(3):
        score, p1_won = simulate_set()
        sets.append(score)
        p1_sets += p1_won
        if p1_sets == 2 or (3 - p1_sets) == 2:
            break
    return sets, f"{p1_sets}-{len(sets) - p1_sets}"

# Predict match
def predict_match(p1_name, p2_name, match_date, xgb, rf, df, player_mapping, player_stats):
    start = time.time()
    p1_id = player_mapping.get(p1_name)
    p2_id = player_mapping.get(p2_name)
    if not p1_id or not p2_id:
        print("Player not found!")
        return None, 0, {}
    
    p1_stats = player_stats.get(p1_id, {k: 50 for k in ["First serve Win %", "Second serve Win %", "First serve return Win %", "Second serve return Win %", "Fatigue", "Clutch %", "Momentum", "Break Conversion %", "Surface Strength", "Tiebreak Wins"]})
    p2_stats = player_stats.get(p2_id, {k: 50 for k in ["First serve Win %", "Second serve Win %", "First serve return Win %", "Second serve return Win %", "Fatigue", "Clutch %", "Momentum", "Break Conversion %", "Surface Strength", "Tiebreak Wins"]})
    
    input_data = pd.DataFrame([{
        "P1 First serve Win %": p1_stats["First serve Win %"],
        "P2 First serve Win %": p2_stats["First serve Win %"],
        "P1 Second serve Win %": p1_stats["Second serve Win %"],
        "P2 Second serve Win %": p2_stats["Second serve Win %"],
        "P1 First serve return Win %": p1_stats["First serve return Win %"],
        "P2 First serve return Win %": p2_stats["First serve return Win %"],
        "P1 Second serve return Win %": p1_stats["Second serve return Win %"],
        "P2 Second serve return Win %": p2_stats["Second serve return Win %"],
        "P1 Fatigue": p1_stats["Fatigue"],
        "P2 Fatigue": p2_stats["Fatigue"],
        "P1 Clutch %": p1_stats["Clutch %"],
        "P2 Clutch %": p2_stats["Clutch %"],
        "P1 Momentum": p1_stats["Momentum"],
        "P2 Momentum": p2_stats["Momentum"],
        "P1 Tiebreak Wins": p1_stats["Tiebreak Wins"],
        "P2 Tiebreak Wins": p2_stats["Tiebreak Wins"],
        "P1 Surface Strength": p1_stats["Surface Strength"],
        "P2 Surface Strength": p2_stats["Surface Strength"],
        "P1 Break Conversion %": p1_stats["Break Conversion %"],
        "P2 Break Conversion %": p2_stats["Break Conversion %"],
        "P1 H2H Strength": 0.5
    }])
    
    xgb_prob = xgb.predict_proba(input_data)[0]
    rf_prob = rf.predict_proba(input_data)[0]
    ensemble_prob = (xgb_prob + rf_prob) / 2
    winner = p1_name if ensemble_prob[1] > ensemble_prob[0] else p2_name
    confidence = max(ensemble_prob)
    sets, match_score = generate_dynamic_scenario(p1_stats, p2_stats)
    
    conn = sqlite3.connect(DB_FILE_NEW)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (player1, player2, match_date, predicted_winner, confidence, predicted_sets, predicted_scores, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (p1_name, p2_name, match_date, winner, confidence, match_score, ", ".join(sets), str(datetime.now())))
    conn.commit()
    conn.close()
    
    print(f"Prediction took {time.time() - start:.2f}s")
    return winner, confidence, {"sets": match_score, "scores": sets}

# Main
if __name__ == "__main__":
    total_start = time.time()
    initialize_db()
    
    # Load data
    print("Loading data...")
    start = time.time()
    df = pd.read_csv(DATA_FILE)
    raw_df = pd.read_csv(RAW_DATA_FILE)
    player_mapping = {name: i for i, name in enumerate(pd.concat([raw_df["Player 1"], raw_df["Player 2"]]).unique())}
    print(f"Data loaded in {time.time() - start:.2f}s")
    
    # Stats and features
    player_stats = load_or_compute_stats(df, raw_df)
    df = add_features(df, player_stats)
    
    # Load or train models
    if os.path.exists(MODEL_XGB_FILE) and os.path.exists(MODEL_RF_FILE):
        xgb = XGBClassifier()
        xgb.load_model(MODEL_XGB_FILE)
        with open(MODEL_RF_FILE, 'rb') as f:
            rf = pickle.load(f)
        print("Loaded saved models.")
    else:
        xgb, rf = train_model(df)
    
    # Command-line interface
    print("Commands: predict <p1> <p2> <date>, retrain, exit")
    while True:
        cmd = input("Enter command: ").strip().split()
        if not cmd:
            continue
        if cmd[0] == "exit":
            break
        elif cmd[0] == "predict" and len(cmd) == 4:
            p1, p2, date = cmd[1], cmd[2], cmd[3]
            winner, conf, scenario = predict_match(p1, p2, date, xgb, rf, df, player_mapping, player_stats)
            if winner:
                print(f"Winner: {winner} ({conf:.2%}), Sets: {scenario['sets']}, Scores: {', '.join(scenario['scores'])}")
        elif cmd[0] == "retrain":
            xgb, rf = train_model(df)
            print("Models retrained.")
        else:
            print("Invalid command. Use: predict <p1> <p2> <date>, retrain, exit")
    
    print(f"Total runtime: {time.time() - total_start:.2f}s")
