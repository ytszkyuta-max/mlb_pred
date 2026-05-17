import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

TARGET_COL = "pitch_type"

PITCH_LABEL_MAP = {
    "FF": "速球(4シーム)",
    "SI": "シンカー",
    "SL": "スライダー",
    "CH": "チェンジアップ",
    "CU": "カーブ",
    "FC": "カットボール",
    "FS": "スプリット",
    "ST": "スイーパー",
    "KC": "ナックルカーブ",
}


def compute_pitcher_mix(df: pd.DataFrame, pitch_types: list) -> dict:
    """
    投手ごとの球種比率を計算してdictで返す。
    {pitcher_id: {FF: 0.45, SL: 0.25, ...}, "__league_avg__": {...}}

    __league_avg__ は未知投手へのフォールバック用。
    生のpitcher IDエンコードと違い、年をまたいでも意味が変わらない。
    """
    counts = (
        df.groupby(["pitcher", TARGET_COL])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=pitch_types, fill_value=0)
    )
    totals = counts.sum(axis=1).replace(0, 1)
    pct = counts.div(totals, axis=0)

    league_avg = pct.mean().to_dict()
    mix = pct.to_dict(orient="index")
    mix["__league_avg__"] = league_avg
    return mix


def apply_pitcher_mix(df: pd.DataFrame, pitch_types: list, pitcher_mix: dict) -> pd.DataFrame:
    """
    pitcher_mix dict を各投球行に結合する。
    未知投手（cross-year で登場した新人など）はリーグ平均で補完。
    """
    league_avg = pitcher_mix["__league_avg__"]
    for pt in pitch_types:
        col = f"pitcher_{pt.lower()}_pct"
        df[col] = df["pitcher"].map(
            lambda pid, pt=pt: pitcher_mix.get(pid, league_avg).get(pt, league_avg[pt])
        )
    return df


def engineer(df: pd.DataFrame, pitch_types: list, feature_cols: list,
             pitcher_mix: dict | None = None):
    """
    特徴量エンジニアリング。

    pitcher_mix=None  → このdfから投手球種比率を計算（Train用）
    pitcher_mix=dict  → 事前計算済みを適用（Test / cross-year用）

    戻り値: (加工済みdf, le_pitcher, pitcher_mix)
    pitcher_mix は推論時に Test dfへ適用するために必要。
    """
    df = df.copy()
    df = df[df[TARGET_COL].isin(pitch_types)].copy()

    for col in ["on_1b", "on_2b", "on_3b"]:
        df[col] = df[col].notna().astype(int)
    df["stand"]    = (df["stand"]    == "R").astype(int)
    df["p_throws"] = (df["p_throws"] == "R").astype(int)

    # 投手IDエンコード（same-period モデルで残す。cross-year では -1 になるが
    # pitcher_mix 特徴量がそれを補う役割を担う）
    le_pitcher = LabelEncoder()
    df["pitcher_enc"] = le_pitcher.fit_transform(df["pitcher"].astype(str))

    # ─── 投手球種比率（pitcher_enc の代替・補完） ─────────────────────────────
    if pitcher_mix is None:
        pitcher_mix = compute_pitcher_mix(df, pitch_types)
    df = apply_pitcher_mix(df, pitch_types, pitcher_mix)

    # ─── ソート ──────────────────────────────────────────────────────────────
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])

    # ─── 打席内ラグ ──────────────────────────────────────────────────────────
    grp_ab = df.groupby(["game_pk", "at_bat_number"])
    df["prev_pitch_type"]  = grp_ab[TARGET_COL].shift(1)
    df["prev2_pitch_type"] = grp_ab[TARGET_COL].shift(2)
    df["prev3_pitch_type"] = grp_ab[TARGET_COL].shift(3)

    for col, new_name in [
        ("release_speed",      "prev_release_speed"),
        ("release_spin_rate",  "prev_release_spin_rate"),
        ("pfx_x",  "prev_pfx_x"),  ("pfx_z",  "prev_pfx_z"),
        ("plate_x","prev_plate_x"),("plate_z","prev_plate_z"),
    ]:
        df[new_name] = grp_ab[col].shift(1).fillna(df[col].median())

    none_label = "NONE"
    prev_map = {p: i for i, p in enumerate([none_label] + list(pitch_types))}
    for lag_col in ["prev_pitch_type", "prev2_pitch_type", "prev3_pitch_type"]:
        df[lag_col] = df[lag_col].fillna(none_label).map(prev_map).fillna(0).astype(int)

    # ─── ゲーム内累積 ─────────────────────────────────────────────────────────
    grp_gp = df.groupby(["game_pk", "pitcher"])
    df["pitch_count_game"]    = grp_gp.cumcount()
    df["pa_count_game"]       = df.groupby(["game_pk", "batter"]).cumcount()
    df["batter_seen_pitcher"] = df.groupby(["game_pk","pitcher","batter"]).cumcount().gt(0).astype(int)

    # ─── ローリング平均 ───────────────────────────────────────────────────────
    df["rolling_release_speed"] = (
        grp_gp["release_speed"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(df["release_speed"].median())
    )
    df["rolling_spin_rate"] = (
        grp_gp["release_spin_rate"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(df["release_spin_rate"].median())
    )

    # ─── ゲーム状況 ───────────────────────────────────────────────────────────
    if "bat_score" in df.columns and "fld_score" in df.columns:
        df["score_diff"] = df["bat_score"] - df["fld_score"]
    else:
        df["score_diff"] = 0
    df["leverage_proxy"]  = df["score_diff"].abs() / (df["inning"] + 1)
    df["count_pressure"]  = df["balls"] - df["strikes"]
    df["runners_on_base"] = df[["on_1b","on_2b","on_3b"]].sum(axis=1)
    df["two_strikes"]     = (df["strikes"] >= 2).astype(int)
    df["full_count"]      = ((df["balls"]==3)&(df["strikes"]==2)).astype(int)
    df["first_pitch"]     = (df["pitch_number"]==1).astype(int)
    df["early_inning"]    = (df["inning"] <= 3).astype(int)
    df["late_inning"]     = (df["inning"] >= 7).astype(int)

    df = df.dropna(subset=feature_cols + [TARGET_COL])
    return df, le_pitcher, pitcher_mix


def encode_target(df: pd.DataFrame):
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    return y, le
