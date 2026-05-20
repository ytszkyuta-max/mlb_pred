# mlb_pred — MLB Pitch Type Predictor

**A data-driven AI that predicts the next pitch type in MLB using Temporal Fusion Transformer + ensemble learning on Statcast data.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[日本語版 README はこちら](README.ja.md)

---

## Overview

A multi-class classification model that predicts the next pitch type using **only pre-pitch game situation data** from MLB Statcast.

- Predicts: FF / SI / SL / CH / CU / FC / FS / ST (**8 pitch types**)
- Accuracy: **46.02%** (within-year) / **43.56%** (cross-year: Train 2023 → Test 2024)
- ~3.5× better than a random baseline of ~12.5% (8-class uniform)
- Key design: Post-pitch physics (velocity, movement) are intentionally excluded to ensure true pre-pitch prediction

```
Game situation (count, runners, previous pitches, pitcher tendencies, ...)
    ↓
XGBoost + CatBoost + Temporal Fusion Transformer
    ↓
Meta-learner (Logistic Regression stacking)
    ↓
Next pitch type with calibrated probability estimates
```

---

## Results

### Within-Year Evaluation (time-series split)

| Model | Test Accuracy |
|---|---|
| XGBoost (standalone) | 37.84% |
| CatBoost (standalone) | 36.84% |
| TFT v4 (standalone) | 38.67% |
| **Ensemble Stack (XGB + CatBoost + TFT)** | **46.02%** |

### Cross-Year Evaluation (Train: 2023 → Test: 2024)

| Model | Test Accuracy |
|---|---|
| XGBoost (standalone) | 36.09% |
| CatBoost (standalone) | 35.41% |
| **TFT v4 (standalone)** | **43.44%** |
| Ensemble Stack | 43.56% |

TFT v4 generalizes best across years. XGBoost and CatBoost degrade more significantly on out-of-year data.

### Count-Based Accuracy (cross-year)

| Count | Accuracy | n |
|---|---|---|
| Batter-favored (3-0) | **77.98%** | 3,483 |
| Full count (3-2) | 48.40% | 18,444 |
| Pitcher-favored (0-2) | 44.39% | 24,789 |
| First pitch (0-0) | 43.17% | 93,477 |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# XGBoost standalone
python3 main.py --mode xgb

# Temporal Fusion Transformer
python3 main.py --mode tft

# Ensemble (XGB + CatBoost + stacking)
python3 main.py --mode ensemble

# Launch interactive prediction UI
streamlit run app.py
```

---

## Features

Only information available **before** the pitch is used (42 features).

| Category | Features |
|---|---|
| Count | balls, strikes, outs_when_up, inning, pitch_number |
| Game situation | score_diff, leverage_proxy, count_pressure, runners_on_base |
| Batter / Pitcher | stand (batter side), p_throws (pitcher arm) |
| Runner situation | on_1b, on_2b, on_3b |
| Pitcher tendencies | pitcher_ff_pct, pitcher_sl_pct, ... (usage rate per pitch type) |
| Lag features | prev_pitch_type (last 3 pitches), prev_release_speed, rolling_spin_rate |

> `release_speed`, `pfx_x/z`, and other post-pitch physics are excluded to prevent data leakage.

---

## Architecture

```
pybaseball (Statcast API)
    ↓ Data fetching + caching (parquet)
Feature engineering
    ├─ Pitcher mix ratios (pitcher_mix)
    ├─ Lag features (shift(1–3) by game_pk × at_bat_number)
    └─ Time-series split (no leakage)
        ↓
    XGBoost ──────────────────────────┐
    CatBoost ─────────────────────────┤→ LogisticRegression → Final prediction
    TFT v4 (GRN + VSN + MHA) ────────┘   (stacking)
        ↓
    Temperature Scaling (probability calibration)
```

**TFT v4** (`src/tft_model.py`):
- Gated Residual Networks (GRN) for feature selection
- Seq → LSTM with BOS token → Multi-head Self-Attention
- Static covariate encoder for game context
- d_model=128, dropout=0.2, CosineAnnealingWarmRestarts

---

## Project Structure

```
mlb_pred/
├── main.py              # Entry point (--mode xgb/tft/hybrid/ensemble)
├── app.py               # Streamlit web app
├── config.yaml          # Data range, features, model settings
├── requirements.txt
└── src/
    ├── data_loader.py   # Statcast data fetching + parquet cache
    ├── features.py      # Feature engineering
    ├── model.py         # Hybrid LSTM model
    ├── tft_model.py     # Temporal Fusion Transformer
    ├── dataset.py       # AtBatDataset (sequence dataset)
    ├── train.py         # Training loop + TemperatureScaler
    └── evaluate.py      # Accuracy + count-based subgroup analysis
```

---

## Roadmap

- [ ] Expand training to multiple seasons (2021–2024) for better cross-year generalization
- [ ] Focus on TFT-only pipeline (XGB/CatBoost add little in cross-year setting)
- [ ] Add pitch location prediction (in/out, high/low)
- [ ] NPB (Nippon Professional Baseball) version
- [ ] FastAPI inference endpoint

---

## Data Sources

- [pybaseball](https://github.com/jldbc/pybaseball) — MLB Statcast data access
- [Baseball Savant](https://baseballsavant.mlb.com/) — Official Statcast source

---

## License

MIT
