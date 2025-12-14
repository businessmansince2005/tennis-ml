# Tennis ML (cleaned)

This repository contains scripts and data used for an offline tennis match prediction model.

Structure:
- src/: main Python scripts (cleaned and path-resilient)
- data/: CSVs and small data files (can stay at repo root; scripts will detect either)
- models/: saved model artifacts and caches

Quick start:
1. Create a Python environment and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the main script (example):

```bash
python src/mc.py
```

3. To push to GitHub:

```bash
git init
git add .
git commit -m "Initial cleaned project"
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```
