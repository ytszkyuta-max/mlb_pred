# mlb_pred — MLB投球予測AI

**MLB Statcastデータ × Temporal Fusion Transformer + アンサンブル学習で8球種を予測するデータ駆動型AI。**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[English README](README.md)

---

## 概要

MLB Statcast データを使い、投球前の**試合状況だけ**から次の球種を予測する多クラス分類モデル。

- 予測対象: FF / SI / SL / CH / CU / FC / FS / ST の **8球種**
- 精度: **46.02%**（同一期間評価）/ **43.56%**（クロスイヤー: Train 2023 → Test 2024）
- ランダムベースライン ~12.5%（8クラス均等）に対して約3.5倍
- 特徴: 投球後にしか得られない物理量（球速・変化量）は意図的に除外し、真の「事前予測」を実現

```
試合状況（カウント・ランナー・直前球種・投手傾向など）
    ↓
XGBoost + CatBoost + Temporal Fusion Transformer
    ↓
LogisticRegression メタ学習器（スタッキング）
    ↓
次の球種を確率付きで予測
```

---

## 精度

### 同一期間評価（Train/Val/Test を時系列分割）

| モデル | テスト精度 |
|---|---|
| XGBoost 単体 | 37.84% |
| CatBoost 単体 | 36.84% |
| TFT v4 単体 | 38.67% |
| **Ensemble Stack（XGB + CatBoost + TFT）** | **46.02%** |

### クロスイヤー評価（Train: 2023 → Test: 2024）

| モデル | テスト精度 |
|---|---|
| XGBoost 単体 | 36.09% |
| CatBoost 単体 | 35.41% |
| **TFT v4 単体** | **43.44%** |
| Ensemble Stack | 43.56% |

クロスイヤー評価では TFT v4 が最も汎化性能が高く、XGBoost・CatBoost との差が顕著。

### カウント別精度（クロスイヤー評価）

| カウント | 精度 | サンプル数 |
|---|---|---|
| 打者有利 (3-0) | **77.98%** | 3,483 |
| フルカウント (3-2) | 48.40% | 18,444 |
| 投手有利 (0-2) | 44.39% | 24,789 |
| 初球 (0-0) | 43.17% | 93,477 |

---

## Quick Start

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# XGBoostモデル単体
python3 main.py --mode xgb

# Temporal Fusion Transformer
python3 main.py --mode tft

# アンサンブルモデル（XGB + CatBoost + スタッキング）
python3 main.py --mode ensemble

# ブラウザで対話予測UI
streamlit run app.py
```

---

## 使用特徴量

投球「前」に得られる情報のみを使用（42特徴量）。

| カテゴリ | 特徴量 |
|---|---|
| カウント | balls, strikes, outs_when_up, inning, pitch_number |
| 試合状況 | score_diff, leverage_proxy, count_pressure, runners_on_base |
| 打者/投手属性 | stand（打席方向）, p_throws（投球腕） |
| ランナー状況 | on_1b, on_2b, on_3b |
| 投手球種傾向 | pitcher_ff_pct, pitcher_sl_pct, ... （9球種の使用率） |
| ラグ特徴量 | prev_pitch_type（直前3球の球種）, prev_release_speed, rolling_spin_rate |

> `release_speed`, `pfx_x/z` などの物理量は除外済み（投球後情報のためリーク扱い）。

---

## アーキテクチャ

```
pybaseball（Statcast API）
    ↓ データ取得・キャッシュ（parquet）
特徴量エンジニアリング
    ├─ 投手球種比率（pitcher_mix）
    ├─ ラグ特徴量（shift(1〜3) by game_pk × at_bat_number）
    └─ 時系列分割（データリーク防止）
        ↓
    XGBoost ─────────────────────┐
    CatBoost ────────────────────┤→ LogisticRegression → 最終予測
    TFT v4（GRN + VSN + MHA）───┘   （スタッキング）
        ↓
    Temperature Scaling（確率較正）
```

**TFT v4 構成** (`src/tft_model.py`):
- Gated Residual Network（GRN）による特徴量選択
- Seq → BOS付きLSTM → Multi-head Self-Attention
- Static covariate encoder で試合状況を文脈化
- d_model=128, dropout=0.2, CosineAnnealingWarmRestarts

---

## ディレクトリ構成

```
mlb_pred/
├── main.py              # エントリーポイント（--mode xgb/tft/hybrid/ensemble）
├── app.py               # Streamlit Webアプリ
├── config.yaml          # データ期間・特徴量・モデル設定
├── requirements.txt
└── src/
    ├── data_loader.py   # Statcastデータ取得 + parquetキャッシュ
    ├── features.py      # 特徴量エンジニアリング
    ├── model.py         # Hybrid LSTM モデル定義
    ├── tft_model.py     # Temporal Fusion Transformer 定義
    ├── dataset.py       # AtBatDataset（系列データセット）
    ├── train.py         # 学習ループ + TemperatureScaler
    └── evaluate.py      # 精度評価 + カウント別分析
```

---

## 今後の拡張予定

- [ ] 複数シーズン（2021〜2024年）への拡張でクロスイヤー精度向上
- [ ] TFT単独路線の強化（XGB/CatBoost依存を削減）
- [ ] コース予測（インコース/アウトコース/高低）の追加
- [ ] NPB（日本プロ野球）版の実装
- [ ] FastAPI による推論エンドポイント

---

## データソース

- [pybaseball](https://github.com/jldbc/pybaseball) — MLB Statcast データ取得
- [Baseball Savant](https://baseballsavant.mlb.com/) — Statcast 公式ソース

---

## License

MIT
