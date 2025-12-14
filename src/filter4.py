import pandas as pd
import json
import time
import re
import sqlite3
import os
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def data_file(name):
    p = DATA_DIR / name
    if p.exists() or not (ROOT / name).exists():
        return str(p)
    return str(ROOT / name)

start_time = time.time()

def split_simple_paired_stats(value):
    if pd.isna(value) or value == "":
        return None, None
    try:
        p1, p2 = map(int, value.split(" - "))
        return p1, p2
    except:
        return None, None

def split_complex_paired_stats(value):
    if pd.isna(value) or value == "":
        return None, None, None, None
    try:
        p1_made, p1_total = map(int, re.match(r"(\d+)/(\d+)", value).groups())
        p2_made, p2_total = map(int, re.search(r" - (\d+)/(\d+)", value).groups())
        return p1_made, p1_total, p2_made, p2_total
    except:
        return None, None, None, None

def extract_set_and_game_scores(json_str):
    if pd.isna(json_str) or json_str == "":
        return {"set_scores": [], "game_scores": [], "match_score": None, "winner": None}
    try:
        data = json.loads(json_str)
        set_scores = []
        game_scores = []
        p1_total_sets = 0
        p2_total_sets = 0
        for set_index, set_data in enumerate(data.get("pointByPoint", []), start=1):
            p1_set_games = 0
            p2_set_games = 0
            set_game_scores = []
            for game in set_data.get("games", []):
                score = game.get("score", {})
                if "homeScore" in score and "awayScore" in score:
                    home_score = score["homeScore"]
                    away_score = score["awayScore"]
                    set_game_scores.append(f"{home_score}-{away_score}")
                    if score.get("scoring") == 1:
                        p1_set_games += 1
                    elif score.get("scoring") == 2:
                        p2_set_games += 1
            set_scores.append(f"{p1_set_games}-{p2_set_games}")
            game_scores.append(set_game_scores)
            if p1_set_games > p2_set_games:
                p1_total_sets += 1
            elif p2_set_games > p1_set_games:
                p2_total_sets += 1
        winner = 1 if p1_total_sets > p2_total_sets else 2 if p2_total_sets > p1_total_sets else None
        return {
            "set_scores": set_scores,
            "game_scores": game_scores,
            "match_score": f"{p1_total_sets}-{p2_total_sets}",
            "winner": winner,
        }
    except Exception as e:
        print(f"Error extracting set and game scores: {e}")
        return {"set_scores": [], "game_scores": [], "match_score": None, "winner": None}

def preprocess_wta_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Initial rows: {len(df)}")
    df = df.drop_duplicates(subset=["Event ID", "Player 1", "Player 2", "Match Date"])
    print(f"Rows after removing duplicates: {len(df)}")

    simple_paired_columns = ["Aces", "Double faults", "Service points won", "Receiver points won", 
                             "Max points in a row", "Service games won", "Max games in a row", 
                             "Return games played", "Break points converted", "Tiebreaks"]
    for col in simple_paired_columns:
        if col in df.columns:
            df[[f"Player 1 {col}", f"Player 2 {col}"]] = df[col].apply(
                lambda x: pd.Series(split_simple_paired_stats(x))
            ).fillna(0)

    complex_paired_columns = {
        "First serve": ("Player 1 First serve Made", "Player 1 First serve Total", "Player 2 First serve Made", "Player 2 First serve Total"),
        "Second serve": ("Player 1 Second serve Made", "Player 1 Second serve Total", "Player 2 Second serve Made", "Player 2 Second serve Total"),
        "First serve points": ("Player 1 First serve points Made", "Player 1 First serve points Total", "Player 2 First serve points Made", "Player 2 First serve points Total"),
        "Second serve points": ("Player 1 Second serve points Made", "Player 1 Second serve points Total", "Player 2 Second serve points Made", "Player 2 Second serve points Total"),
        "Break points saved": ("Player 1 Break points saved Made", "Player 1 Break points saved Total", "Player 2 Break points saved Made", "Player 2 Break points saved Total"),
        "First serve return points": ("Player 1 First serve return points Made", "Player 1 First serve return points Total", "Player 2 First serve return points Made", "Player 2 First serve return points Total"),
        "Second serve return points": ("Player 1 Second serve return points Made", "Player 1 Second serve return points Total", "Player 2 Second serve return points Made", "Player 2 Second serve return points Total")
    }
    for col, (p1_made, p1_total, p2_made, p2_total) in complex_paired_columns.items():
        if col in df.columns:
            df[[p1_made, p1_total, p2_made, p2_total]] = df[col].apply(
                lambda x: pd.Series(split_complex_paired_stats(x))
            ).fillna(0)

    df = df.drop(columns=simple_paired_columns + list(complex_paired_columns.keys()), errors="ignore")

    df["Match Date"] = pd.to_datetime(df["Match Date"])
    df["Year"] = df["Match Date"].dt.year
    df["Month"] = df["Match Date"].dt.month
    df["Day"] = df["Match Date"].dt.day

    print("Extracting set and game scores...")
    score_results = [extract_set_and_game_scores(json_str) for json_str in df["Point-by-point data"]]

    max_sets = 3
    max_games_per_set = 12
    new_columns = {}
    for i in range(max_sets):
        new_columns[f"Set {i + 1}"] = [
            result["set_scores"][i] if i < len(result["set_scores"]) else None for result in score_results
        ]
    for set_index in range(max_sets):
        for game_index in range(max_games_per_set):
            new_columns[f"Set {set_index + 1} Game {game_index + 1}"] = [
                result["game_scores"][set_index][game_index]
                if set_index < len(result["game_scores"]) and game_index < len(result["game_scores"][set_index])
                else None
                for result in score_results
            ]
    new_columns["Match Score"] = [result["match_score"] for result in score_results]
    new_columns["Winner"] = [result["winner"] for result in score_results]
    new_columns["Player 1 Tiebreak Wins"] = [
        sum(1 for score in result["set_scores"] if score == "7-6") for result in score_results
    ]
    new_columns["Player 2 Tiebreak Wins"] = [
        sum(1 for score in result["set_scores"] if score == "6-7") for result in score_results
    ]
    new_columns["Match Duration"] = [
        sum(len(game["points"]) for set_data in json.loads(json_str)["pointByPoint"] for game in set_data["games"])
        if not pd.isna(json_str) else 0 for json_str in df["Point-by-point data"]
    ]

    new_columns_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_columns_df], axis=1)

    df["Player 1 First Serve Win %"] = (df["Player 1 First serve points Made"] / df["Player 1 First serve points Total"]).fillna(0) * 100
    df["Player 2 First Serve Win %"] = (df["Player 2 First serve points Made"] / df["Player 2 First serve points Total"]).fillna(0) * 100
    df["Player 1 Second Serve Win %"] = (df["Player 1 Second serve points Made"] / df["Player 1 Second serve points Total"]).fillna(0) * 100
    df["Player 2 Second Serve Win %"] = (df["Player 2 Second serve points Made"] / df["Player 2 Second serve points Total"]).fillna(0) * 100
    df["Player 1 First Serve Return Win %"] = (df["Player 1 First serve return points Made"] / df["Player 1 First serve return points Total"]).fillna(0) * 100
    df["Player 2 First Serve Return Win %"] = (df["Player 2 First serve return points Made"] / df["Player 2 First serve return points Total"]).fillna(0) * 100
    df["Player 1 Second Serve Return Win %"] = (df["Player 1 Second serve return points Made"] / df["Player 1 Second serve return points Total"]).fillna(0) * 100
    df["Player 2 Second Serve Return Win %"] = (df["Player 2 Second serve return points Made"] / df["Player 2 Second serve return points Total"]).fillna(0) * 100

    all_players = pd.concat([df["Player 1"], df["Player 2"]]).unique()
    player_mapping = {player: i for i, player in enumerate(all_players)}
    df["Player 1 ID"] = df["Player 1"].map(player_mapping)
    df["Player 2 ID"] = df["Player 2"].map(player_mapping)

    df = df.drop(columns=["Player 1", "Player 2", "Match Date", "Point-by-point data", "Total points", "Total games won", "Service games played"], errors="ignore")

    df.fillna(0, inplace=True)
    return df

def create_points_table(filepath):
    print("Creating points table...")
    conn = sqlite3.connect(str(DATA_DIR / "points.db"))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS points (
            "Event ID" TEXT,
            "Set" INTEGER,
            "Game" INTEGER,
            "Point" INTEGER,
            "Server" INTEGER,
            "Winner" INTEGER,
            "Point Type" TEXT
        )
    """)
    df = pd.read_csv(filepath)
    points_data = []
    error_count = 0
    for _, row in df.iterrows():
        event_id = row["Event ID"]
        point_data = row["Point-by-point data"]
        if pd.isna(point_data):
            continue
        try:
            data = json.loads(point_data)
            point_by_point = data.get("pointByPoint", [])
            if not point_by_point:
                continue
            for set_index, set_data in enumerate(point_by_point, start=1):
                games = set_data.get("games", [])
                if not games:
                    continue
                for game_index, game_data in enumerate(games, start=1):
                    points = game_data.get("points", [])
                    if not points:
                        continue
                    score = game_data.get("score", {})
                    server = score.get("serving", 0)
                    winner = score.get("scoring", 0)
                    for point_index, point in enumerate(points, start=1):
                        point_type = point.get("homePointType" if server == 1 else "awayPointType", "NA")
                        points_data.append((event_id, set_index, game_index, point_index, server, winner, point_type))
                        cursor.execute("""
                            INSERT INTO points ("Event ID", "Set", "Game", "Point", "Server", "Winner", "Point Type")
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (event_id, set_index, game_index, point_index, server, winner, point_type))
        except Exception as e:
            error_count += 1
            print(f"Error processing Event ID {event_id}: {e}")

    # Debug
    points_df = pd.DataFrame(points_data, columns=["Event ID", "Set", "Game", "Point", "Server", "Winner", "Point Type"])
    print("Points data sample before saving to points.db:")
    print(points_df.head())
    print(f"Total points extracted: {len(points_df)}")
    print(f"Total JSON errors encountered: {error_count}")

    conn.commit()
    conn.close()
    print("Points table created successfully.")

if __name__ == "__main__":
    input_file = data_file("n.csv")
    output_file = str(DATA_DIR / "ml_ready4.csv")
    ml_ready_df = preprocess_wta_data(input_file)
    if os.path.exists(output_file):
        ml_ready_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        ml_ready_df.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to '{output_file}'")
    print("\nDataset Info:")
    print(ml_ready_df.info())
    print("\nFirst 5 Rows:")
    print(ml_ready_df.head())
    create_points_table(input_file)
    end_time = time.time()
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
