# mlb_pred — MLB投球予測AI

**MLB Statcastデータ×アンサンブル学習で9球種を43.85%の精度で予測するデータ駆動型AI。**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[English README](README.md)

---

## 概要

MLB Statcast データを使い、投球前の**試合状況だけ**から次の球種を予測する多クラス分類モデル。

- 予測対象: FF / SI / SL / CH / CU / FC / FS / ST / KC の **9球種**
- 精度: **43.85%**（ランダム比較 ~11%、約4倍）
- 特徴: 投球後にしか得られない物理量（球速・変化量）は意図的に除外し、真の「事前予測」を実現

```
試合状況（カウント・ランナー・直前球種・投手IDなど）
    ↓
XGBoost + CatBoost スタッキング
    ↓
次の球種を確率付きで予測
```

---

## 精度

| モデル | テスト精度 |
|---|---|
| XGBoost 単体 | 36.10% |
| CatBoost 単体 | 36.65% |
| **Ensemble Stack（XGB + CatBoost + Logistic Regression）** | **43.85%** |

ランダムベースライン ~11%（9クラス均等）に対して約4倍。

---

## Quick Start

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# XGBoostモデルで学習・評価
python3 main.py --mode xgb

# アンサンブルモデル（推奨）
python3 main.py --mode ensemble

# ブラウザで対話予測UI
streamlit run app.py
```

---

## 使用特徴量

投球「前」に得られる情報のみを使用。

| カテゴリ | 特徴量 |
|---|---|
| カウント | balls, strikes, outs_when_up, inning |
| 打者/投手属性 | stand（打席方向）, p_throws（投球腕）, pitcher_enc |
| ランナー状況 | on_1b, on_2b, on_3b |
| ラグ特徴量 | prev_pitch_type（直前球種）, prev_release_speed（直前球速） |

> `release_speed`, `pfx_x/z` などの物理量は除外済み（投球後情報のためリーク扱い）。

---

## アーキテクチャ

```
pybaseball（Statcast API）
    ↓ データ取得・キャッシュ（parquet）
特徴量エンジニアリング
    ├─ 投手エンコーディング（LabelEncoder）
    ├─ ラグ特徴量（shift(1) by game_pk × at_bat_number）
    └─ 時系列分割（train 80% / val 10% / test 10%）
        ↓
    XGBoost ────┐
    CatBoost ───┤ → メタ学習器（Logistic Regression）→ 最終予測
```

---

## ディレクトリ構成

```
mlb_pred/
├── main.py              # エントリーポイント
├── app.py               # Streamlit Webアプリ
├── config.yaml          # データ期間・特徴量・モデル設定
├── requirements.txt
└── src/
    ├── data_loader.py   # Statcastデータ取得 + parquetキャッシュ
    ├── features.py      # 特徴量エンジニアリング
    ├── train.py         # 時系列分割 + 学習
    └── evaluate.py      # 精度評価 + カウント別分析
```

---

## 今後の拡張予定

- [ ] 複数シーズン（2020〜2024年）への拡張
- [ ] コース予測（インコース/アウトコース/高低）の追加
- [ ] NPB（日本プロ野球）版の実装
- [ ] FastAPI による推論エンドポイント
- [ ] Optuna によるハイパーパラメータ最適化

---

## データソース

- [pybaseball](https://github.com/jldbc/pybaseball) — MLB Statcast データ取得
- [Baseball Savant](https://baseballsavant.mlb.com/) — Statcast 公式ソース

---

## License

MIT
