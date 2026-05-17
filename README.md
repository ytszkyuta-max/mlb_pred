# mlb_pred — MLB Pitch Type Predictor

**A data-driven AI that predicts the next pitch type in MLB. Achieves 43.85% accuracy across 9 pitch types using Statcast data and ensemble learning.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[日本語版 README はこちら](README.ja.md)

---

## Overview

A multi-class classification model that predicts the next pitch type using **only pre-pitch game situation data** from MLB Statcast.

- Predicts: FF / SI / SL / CH / CU / FC / FS / ST / KC (**9 pitch types**)
- Accuracy: **43.85%** (vs. random baseline ~11% — approximately 4×)
- Key design: Post-pitch physics (velocity, movement) are intentionally excluded to ensure true pre-pitch prediction

```
Game situation (count, runners, previous pitch, pitcher ID, ...)
    ↓
XGBoost + CatBoost stacking
    ↓
Next pitch type with probability estimates
```

---

## Results

| Model | Test Accuracy |
|---|---|
| XGBoost (standalone) | 36.10% |
| CatBoost (standalone) | 36.65% |
| **Ensemble Stack (XGB + CatBoost + Logistic Regression)** | **43.85%** |

~4× better than a random baseline of ~11% (9-class uniform).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train and evaluate with XGBoost
python3 main.py --mode xgb

# Train and evaluate with ensemble (recommended)
python3 main.py --mode ensemble

# Launch interactive prediction UI
streamlit run app.py
```

---

## Features

Only information available **before** the pitch is used.

| Category | Features |
|---|---|
| Count | balls, strikes, outs_when_up, inning |
| Batter / Pitcher | stand (batter side), p_throws (pitcher arm), pitcher_enc |
| Runner situation | on_1b, on_2b, on_3b |
| Lag features | prev_pitch_type, prev_release_speed |

> `release_speed`, `pfx_x/z`, and other post-pitch physics are excluded to prevent data leakage.

---

## Architecture

```
pybaseball (Statcast API)
    ↓ Data fetching + caching (parquet)
Feature engineering
    ├─ Pitcher encoding (LabelEncoder)
    ├─ Lag features (shift(1) by game_pk × at_bat_number)
    └─ Time-series split (train 80% / val 10% / test 10%)
        ↓
    XGBoost ────┐
    CatBoost ───┤ → Meta-learner (Logistic Regression) → Final prediction
```

---

## Project Structure

```
mlb_pred/
├── main.py              # Entry point
├── app.py               # Streamlit web app
├── config.yaml          # Data range, features, model settings
├── requirements.txt
└── src/
    ├── data_loader.py   # Statcast data fetching + parquet cache
    ├── features.py      # Feature engineering
    ├── train.py         # Time-series split + training
    └── evaluate.py      # Accuracy + count-based subgroup analysis
```

---

## Roadmap

- [ ] Expand to multiple seasons (2020–2024)
- [ ] Add pitch location prediction (in/out, high/low)
- [ ] NPB (Nippon Professional Baseball) version
- [ ] FastAPI inference endpoint
- [ ] Hyperparameter optimization with Optuna

---

## Data Sources

- [pybaseball](https://github.com/jldbc/pybaseball) — MLB Statcast data access
- [Baseball Savant](https://baseballsavant.mlb.com/) — Official Statcast source

---

## License

MIT
