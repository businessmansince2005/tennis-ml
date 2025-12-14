To: jmkchowdaryjonnalagadda@gmail.com
Subject: Demonstration of Tennis ML model & access to full dataset

Hi,

I'm sharing a short summary of the model we've developed and how you can verify and reproduce our evaluation artifacts.

Key items included in the repository:
- MODEL_CARD.md — model provenance, training details, SHA256, metrics and reproducibility steps.
- data/sample.csv — an anonymized sample used for public evaluation (n=195).
- results/metrics.json, results/roc.png, results/confusion.png — evaluation outputs from the sample.
- scripts/evaluate_sample.py — script used to prepare sample features and compute metrics.

Snapshot (public evaluation):
- Accuracy: 0.51795
- ROC AUC: 0.55468
- Sample size: 195
- Model artifact: models/super_tennis_xgb_v2.json
- Model SHA256: 8a5859cfd2462f3806e2568d1d2797715247fbd48407f25e231186f5712e6a42
- Commit: 4d39d40 (2025-12-14)

How to verify locally (quick):
1) Clone the repo and confirm commit: `git rev-parse --short HEAD`
2) Verify model checksum (PowerShell): `Get-FileHash models/super_tennis_xgb_v2.json -Algorithm SHA256`
3) Re-run evaluation: `python -m venv venv && venv\Scripts\python -m pip install -r requirements.txt && python scripts/evaluate_sample.py`

If you'd like access to the full dataset or model provenance artifacts, please reply to this email and we can start the DUA/NDA process (see DATA_ACCESS.md for the workflow). I'm happy to schedule a short call to walk through results and reproduce training under your supervision.

Best regards,
[Your Team]
