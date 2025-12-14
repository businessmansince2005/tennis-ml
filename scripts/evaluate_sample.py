#!/usr/bin/env python3
"""Evaluate model on the anonymized sample and save metrics and plots to `results/`."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def load_model(path):
    model = XGBClassifier()
    model.load_model(str(path))
    return model


def main(sample='data/sample.csv'):
    sample_path = ROOT / sample
    df = pd.read_csv(sample_path)
    # Ensure derived features expected by the model exist on the sample
    def prepare_features(df):
        feature_cols = ["Serve Win %", "Return Win %", "Recent Form", "Momentum", "Fatigue",
                        "Clutch %", "Consistency", "Break Point Edge", "Max Streak", "Ace Rate", "Double Fault Rate"]

        # Helper getters
        def pct(made_col, total_col):
            if made_col in df.columns and total_col in df.columns:
                return pd.to_numeric(df[made_col], errors='coerce') / pd.to_numeric(df[total_col], errors='coerce') * 100
            return pd.Series([None] * len(df))

        for side in ['1', '2']:
            prefix = f'P{side} '
            # Serve Win %: prefer 'Player X First serve Win %' else compute from points
            srv_col = f'Player {side} First serve Win %'
            if f'{prefix}Serve Win %' not in df.columns:
                if srv_col in df.columns:
                    df[f'{prefix}Serve Win %'] = pd.to_numeric(df[srv_col], errors='coerce')
                else:
                    df[f'{prefix}Serve Win %'] = pct(f'Player {side} First serve points Made', f'Player {side} First serve points Total')

            # Return Win %: map from 'Player X First serve return Win %'
            ret_col = f'Player {side} First serve return Win %'
            if f'{prefix}Return Win %' not in df.columns:
                if ret_col in df.columns:
                    df[f'{prefix}Return Win %'] = pd.to_numeric(df[ret_col], errors='coerce')
                else:
                    df[f'{prefix}Return Win %'] = 0

            # Fatigue from match duration (minutes)
            if f'{prefix}Fatigue' not in df.columns:
                if 'Match Duration' in df.columns:
                    df[f'{prefix}Fatigue'] = pd.to_numeric(df['Match Duration'], errors='coerce') / 60.0
                else:
                    df[f'{prefix}Fatigue'] = 0

            # Tiebreak Wins
            tb_col = f'Player {side} Tiebreak Wins'
            if f'{prefix}Tiebreak Wins' not in df.columns:
                df[f'{prefix}Tiebreak Wins'] = pd.to_numeric(df[tb_col], errors='coerce') if tb_col in df.columns else 0

            # Break Conversion %: map if we have converted count
            bc_col = f'Player {side} Break points converted'
            if f'{prefix}Break Conversion %' not in df.columns:
                df[f'{prefix}Break Conversion %'] = pd.to_numeric(df[bc_col], errors='coerce') if bc_col in df.columns else 0

            # Max Streak (use points or games)
            ms_col = f'Player {side} Max points in a row'
            if f'{prefix}Max Streak' not in df.columns:
                df[f'{prefix}Max Streak'] = pd.to_numeric(df[ms_col], errors='coerce') if ms_col in df.columns else 0

            # Ace Rate and Double Fault Rate (percent)
            if f'{prefix}Ace Rate' not in df.columns:
                df[f'{prefix}Ace Rate'] = (pd.to_numeric(df.get(f'Player {side} Aces', 0), errors='coerce') / (pd.to_numeric(df.get(f'Player {side} Service points won', 1), errors='coerce') + 1)) * 100
            if f'{prefix}Double Fault Rate' not in df.columns:
                df[f'{prefix}Double Fault Rate'] = (pd.to_numeric(df.get(f'Player {side} Double faults', 0), errors='coerce') / (pd.to_numeric(df.get(f'Player {side} Service points won', 1), errors='coerce') + 1)) * 100

            # Defaults for other features
            for defc in ['Recent Form', 'Momentum', 'Clutch %', 'Consistency', 'Break Point Edge', 'Surface Strength']:
                colname = f'{prefix}{defc}'
                if colname not in df.columns:
                    df[colname] = 0 if defc not in ['Surface Strength'] else 0.5

        # H2H and cluster defaults
        if 'P1 H2H Strength' not in df.columns:
            df['P1 H2H Strength'] = 0.5
        if 'P1 Cluster' not in df.columns:
            df['P1 Cluster'] = 0
        if 'P2 Cluster' not in df.columns:
            df['P2 Cluster'] = 0

        return df

    df = prepare_features(df)
    # Select the same features used during the model training in mc.py (25 features)
    features = [
        f"P{i} {col}" for i in [1, 2] for col in [
            "Serve Win %", "Return Win %", "Recent Form", "Momentum", "Fatigue",
            "Clutch %", "Consistency", "Break Point Edge", "Max Streak", "Ace Rate", "Double Fault Rate"
        ]
    ] + ["P1 H2H Strength", "P1 Cluster", "P2 Cluster"]
    # Only keep rows with a known winner (1 or 2), similar to training
    df = df[df['Winner'].isin([1, 2])].copy()
    X = df[features].select_dtypes('number').fillna(0)
    y = df['Winner'].map({1: 1, 2: 0})

    model = load_model(MODELS_DIR / 'super_tennis_xgb_v2.json')
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)

    acc = float(accuracy_score(y, preds))
    try:
        auc = float(roc_auc_score(y, probs))
    except Exception:
        auc = None

    cm = confusion_matrix(y, preds).tolist()

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = {'accuracy': acc, 'roc_auc': auc, 'n': int(len(df))}
    (RESULTS_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (RESULTS_DIR / 'confusion_matrix.json').write_text(json.dumps({'cm': cm}))

    # ROC plot
    if auc is not None:
        fpr, tpr, _ = roc_curve(y, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('ROC on Sample')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'roc.png')

    # Confusion matrix heatmap
    plt.figure()
    plt.imshow(np.array(cm), cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.savefig(RESULTS_DIR / 'confusion.png')

    print('Saved metrics to', RESULTS_DIR)


if __name__ == '__main__':
    main()