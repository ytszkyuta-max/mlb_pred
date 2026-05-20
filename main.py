import argparse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import yaml
import joblib

from src.data_loader import load
from src.features import engineer, encode_target
from src.train import (
    time_split, time_split_dfs, train, train_hybrid,
    train_catboost, catboost_proba, train_stacking, train_blend_weights,
    TemperatureScaler,
)
from src.evaluate import evaluate, count_subgroup_analysis, evaluate_hybrid, evaluate_ensemble, model_agreement_analysis, evaluate_blend, evaluate_temperature_scaled

def main():
    parser = argparse.ArgumentParser(description="MLB投球予測AI")
    parser.add_argument(
        "--mode", choices=["xgb", "hybrid", "ensemble", "tft"], default="hybrid",
        help="xgb: XGBoost単体  hybrid: LSTM+全結合ハイブリッド  ensemble: XGBoost+CatBoost+LSTM スタッキング  tft: Temporal Fusion Transformer (デフォルト: hybrid)",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    cfg_path = project_dir / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    def abs_path(p: str) -> Path:
        return project_dir / p

    # 1. データ取得（2回目以降はparquetキャッシュから高速読み込み）
    df_raw = load(
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        cache_path=str(abs_path(cfg["data"]["cache_path"])),
    )

    # 2. 特徴量エンジニアリング
    feature_cols = cfg["features"]["cols"]
    pitch_types = cfg["features"]["pitch_types"]
    df, le_pitcher, pitcher_mix = engineer(df_raw, pitch_types, feature_cols)

    print(f"\n使用データ: {len(df):,} 投球  球種数: {df['pitch_type'].nunique()}")
    print(df["pitch_type"].value_counts().to_string())

    # 3. ターゲットエンコード
    y, le_target = encode_target(df)

    split_cfg = cfg["split"]
    out_dir = abs_path(cfg["output"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "xgb":
        # ── XGBoost ──────────────────────────────────────────────────────────
        X_train, X_val, X_test, y_train, y_val, y_test, df_test = time_split(
            df, feature_cols, y,
            test_ratio=split_cfg["test_ratio"],
            val_ratio=split_cfg["val_ratio"],
        )
        print(f"\nTrain: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

        model = train(X_train, y_train, X_val, y_val, cfg["model"])
        acc, y_pred = evaluate(model, X_test, y_test, le_target, feature_cols)
        count_subgroup_analysis(df_test, y_test, y_pred)

        save_path = out_dir / cfg["output"]["model_name"]
        joblib.dump(
            {"model": model, "le_target": le_target, "le_pitcher": le_pitcher, "pitcher_mix": pitcher_mix},
            save_path,
        )
        print(f"モデル保存: {save_path}")

    elif args.mode == "hybrid":
        # ── Hybrid LSTM ───────────────────────────────────────────────────────
        import torch
        from src.dataset import AtBatDataset, fit_scalers, SEQ_DIM
        from src.model import HybridLSTMClassifier

        # デバイス選択: Apple Silicon MPS → CUDA → CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"\nDevice: {device}")

        # 時系列でDataFrameとして分割
        df_train, df_val, df_test = time_split_dfs(
            df,
            test_ratio=split_cfg["test_ratio"],
            val_ratio=split_cfg["val_ratio"],
        )
        print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

        # スケーラーを訓練データで fit → val/test に適用
        scalers = fit_scalers(df_train, feature_cols)

        ds_train = AtBatDataset(df_train, feature_cols, le_target, scalers)
        ds_val   = AtBatDataset(df_val,   feature_cols, le_target, scalers)
        ds_test  = AtBatDataset(df_test,  feature_cols, le_target, scalers)
        print(f"サンプル数 — Train: {len(ds_train):,}  Val: {len(ds_val):,}  Test: {len(ds_test):,}")

        cfg_hybrid = cfg["hybrid"]
        num_classes = len(le_target.classes_)
        model = HybridLSTMClassifier(
            seq_dim=SEQ_DIM,
            tab_dim=len(feature_cols),
            lstm_hidden=cfg_hybrid["lstm_hidden"],
            num_layers=cfg_hybrid["num_layers"],
            num_classes=num_classes,
            dropout=cfg_hybrid["dropout"],
        ).to(device)

        print(f"\nモデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}\n")
        model = train_hybrid(ds_train, ds_val, model, cfg_hybrid, device)

        acc, y_pred, y_test_np = evaluate_hybrid(model, ds_test, cfg_hybrid, le_target, device)
        count_subgroup_analysis(df_test.reset_index(drop=True), y_test_np, y_pred)

        save_path = out_dir / cfg_hybrid["model_name"]
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_cfg": {
                    "seq_dim": SEQ_DIM,
                    "tab_dim": len(feature_cols),
                    "lstm_hidden": cfg_hybrid["lstm_hidden"],
                    "num_layers": cfg_hybrid["num_layers"],
                    "num_classes": num_classes,
                    "dropout": cfg_hybrid["dropout"],
                },
                "le_target": le_target,
                "le_pitcher": le_pitcher,
                "scalers": scalers,
                "feature_cols": feature_cols,
            },
            save_path,
        )
        print(f"モデル保存: {save_path}")

    elif args.mode == "tft":
        # ── Temporal Fusion Transformer ───────────────────────────────────────
        import torch
        from src.dataset import AtBatDataset, fit_scalers, SEQ_DIM
        from src.tft_model import TFTPitchClassifier

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"\nDevice: {device}")

        df_train, df_val, df_test = time_split_dfs(
            df,
            test_ratio=split_cfg["test_ratio"],
            val_ratio=split_cfg["val_ratio"],
        )
        print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

        scalers = fit_scalers(df_train, feature_cols)
        ds_train = AtBatDataset(df_train, feature_cols, le_target, scalers)
        ds_val   = AtBatDataset(df_val,   feature_cols, le_target, scalers)
        ds_test  = AtBatDataset(df_test,  feature_cols, le_target, scalers)
        print(f"サンプル数 — Train: {len(ds_train):,}  Val: {len(ds_val):,}  Test: {len(ds_test):,}")

        cfg_tft = cfg["tft"]
        num_classes = len(le_target.classes_)
        model = TFTPitchClassifier(
            seq_dim=SEQ_DIM,
            tab_dim=len(feature_cols),
            num_classes=num_classes,
            d_model=cfg_tft["d_model"],
            num_heads=cfg_tft["num_heads"],
            num_lstm_layers=cfg_tft["num_lstm_layers"],
            dropout=cfg_tft["dropout"],
        ).to(device)

        print(f"\nモデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}\n")
        model = train_hybrid(ds_train, ds_val, model, cfg_tft, device)

        acc, y_pred, y_test_np = evaluate_hybrid(model, ds_test, cfg_tft, le_target, device)
        count_subgroup_analysis(df_test.reset_index(drop=True), y_test_np, y_pred)

        save_path = out_dir / cfg_tft["model_name"]
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_cfg": {
                    "seq_dim": SEQ_DIM,
                    "tab_dim": len(feature_cols),
                    "num_classes": num_classes,
                    "d_model": cfg_tft["d_model"],
                    "num_heads": cfg_tft["num_heads"],
                    "num_lstm_layers": cfg_tft["num_lstm_layers"],
                    "dropout": cfg_tft["dropout"],
                },
                "le_target": le_target,
                "le_pitcher": le_pitcher,
                "scalers": scalers,
                "feature_cols": feature_cols,
            },
            save_path,
        )
        print(f"モデル保存: {save_path}")

    else:
        # ── Ensemble Stack ────────────────────────────────────────────────────
        # XGBoost (表形式) + CatBoost (カテゴリ特化) の 2 モデルをベース学習器とし、
        # val 予測確率をメタ特徴として LogisticRegression でスタッキングする。
        # カテゴリ特徴量のインデックスを feature_cols から動的に決定
        cat_features_names = ["stand", "p_throws", "prev_pitch_type", "pitcher_enc"]
        cat_feature_indices = [
            feature_cols.index(f) for f in cat_features_names if f in feature_cols
        ]

        # ── 表形式データ分割 ──────────────────────────────────────────────────
        X_train, X_val, X_test, y_train, y_val, y_test, df_test = time_split(
            df, feature_cols, y,
            test_ratio=split_cfg["test_ratio"],
            val_ratio=split_cfg["val_ratio"],
        )
        print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}\n")

        # ── [1/2] XGBoost ─────────────────────────────────────────────────────
        print("=" * 50)
        print("[1/2] XGBoost 学習中...")
        print("=" * 50)
        xgb_model = train(X_train, y_train, X_val, y_val, cfg["model"])
        xgb_val_probs = xgb_model.predict_proba(X_val)
        xgb_test_probs = xgb_model.predict_proba(X_test)

        # ── [2/2] CatBoost ────────────────────────────────────────────────────
        print("\n" + "=" * 50)
        print("[2/2] CatBoost 学習中...")
        print("=" * 50)
        cat_model = train_catboost(
            X_train, y_train, X_val, y_val, cfg["catboost"], cat_feature_indices
        )
        cat_val_probs = catboost_proba(cat_model, X_val, cat_feature_indices)
        cat_test_probs = catboost_proba(cat_model, X_test, cat_feature_indices)

        # ── 単体精度の確認 ────────────────────────────────────────────────────
        from sklearn.metrics import accuracy_score as _acc
        xgb_test_acc = _acc(y_test, xgb_model.predict(X_test))
        cat_test_acc = _acc(y_test, catboost_proba(cat_model, X_test, cat_feature_indices).argmax(axis=1))
        print(f"\n── 単体テスト精度 ──")
        print(f"  XGBoost  : {xgb_test_acc:.4f} ({xgb_test_acc*100:.2f}%)")
        print(f"  CatBoost : {cat_test_acc:.4f} ({cat_test_acc*100:.2f}%)")

        # ── diversity診断 ─────────────────────────────────────────────────────
        val_probs_list = [xgb_val_probs, cat_val_probs]
        test_probs_list = [xgb_test_probs, cat_test_probs]
        model_agreement_analysis(test_probs_list, ["XGBoost", "CatBoost"])

        # ── スタッキング ──────────────────────────────────────────────────────
        print("=" * 50)
        print("[スタッキング] メタ学習器 (LogisticRegression) 訓練中...")
        print("=" * 50)
        meta_model = train_stacking(
            val_probs_list, y_val, meta_C=cfg["ensemble"].get("meta_C", 0.5)
        )
        stack_acc, stack_pred = evaluate_ensemble(meta_model, test_probs_list, y_test, le_target)

        # ── Temperature Scaling ───────────────────────────────────────────────
        meta_val_proba = meta_model.predict_proba(np.hstack(val_probs_list))
        temp_scaler = TemperatureScaler().fit(meta_val_proba, y_val)
        meta_test_proba = meta_model.predict_proba(np.hstack(test_probs_list))
        evaluate_temperature_scaled(temp_scaler, meta_test_proba, y_test, le_target)

        # ── ブレンディング（Optuna重み最適化）────────────────────────────────
        print("=" * 50)
        print("[ブレンディング] Optuna で重みを最適化中...")
        print("=" * 50)
        blend_weights = train_blend_weights(val_probs_list, y_val, n_trials=cfg["ensemble"].get("blend_trials", 300))
        blend_acc, blend_pred = evaluate_blend(blend_weights, test_probs_list, y_test, le_target)

        # ── 精度サマリ ────────────────────────────────────────────────────────
        best_acc = max(stack_acc, blend_acc)
        y_pred = stack_pred if stack_acc >= blend_acc else blend_pred
        acc = best_acc
        print(f"── 精度サマリ ──")
        print(f"  XGBoost  単体    : {xgb_test_acc*100:.2f}%")
        print(f"  CatBoost 単体    : {cat_test_acc*100:.2f}%")
        print(f"  Ensemble Stack   : {stack_acc*100:.2f}%  ({stack_acc-xgb_test_acc:+.4f} vs XGB)")
        print(f"  Blend (Optuna)   : {blend_acc*100:.2f}%  ({blend_acc-xgb_test_acc:+.4f} vs XGB)  ← 採用" if blend_acc >= stack_acc else f"  Blend (Optuna)   : {blend_acc*100:.2f}%  ({blend_acc-xgb_test_acc:+.4f} vs XGB)")
        print()
        count_subgroup_analysis(df_test, y_test, y_pred)

        # ── 保存 ──────────────────────────────────────────────────────────────
        save_path = out_dir / cfg["ensemble"]["model_name"]
        joblib.dump(
            {
                "meta_model": meta_model,
                "temp_scaler": temp_scaler,
                "xgb_model": xgb_model,
                "cat_model": cat_model,
                "le_target": le_target,
                "le_pitcher": le_pitcher,
                "pitcher_mix": pitcher_mix,
                "feature_cols": feature_cols,
                "cat_feature_indices": cat_feature_indices,
            },
            save_path,
        )
        print(f"モデル保存: {save_path}")


if __name__ == "__main__":
    main()
