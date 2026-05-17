import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from .features import PITCH_LABEL_MAP

# READMEの考察より: カウントによって球種分布が大きく異なるためサブグループ評価が必要
COUNT_GROUPS = {
    "投手有利 (0-2)": {"balls": 0, "strikes": 2},
    "打者有利 (3-0)": {"balls": 3, "strikes": 0},
    "フルカウント (3-2)": {"balls": 3, "strikes": 2},
    "初球 (0-0)": {"balls": 0, "strikes": 0},
    "2ストライク計": None,  # 特殊処理
}


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, le_target, feature_cols: list):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== テスト精度: {acc:.4f} ({acc*100:.2f}%) ===\n")

    labels = le_target.classes_
    names = [PITCH_LABEL_MAP.get(lb, lb) for lb in labels]
    print(classification_report(y_test, y_pred, target_names=names))

    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("── 特徴量重要度 ──")
    print(importance.to_string())
    print()

    return acc, y_pred


def count_subgroup_analysis(df_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray):
    """
    カウント別精度分析。
    READMEの考察: 3-0と0-2では球種分布が大きく異なるため、全体accuracyでは見えない傾向を把握する。
    """
    print("── カウント別精度分析 ──")
    df = df_test.copy().reset_index(drop=True)
    df["y_true"] = y_test
    df["y_pred"] = y_pred
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)

    rows = []
    for name, cond in COUNT_GROUPS.items():
        if name == "2ストライク計":
            mask = df["strikes"] == 2
        else:
            mask = (df["balls"] == cond["balls"]) & (df["strikes"] == cond["strikes"])

        sub = df[mask]
        if len(sub) == 0:
            continue
        acc = sub["correct"].mean()
        rows.append({"カウント": name, "精度": f"{acc:.4f}", "サンプル数": f"{len(sub):,}"})

    print(pd.DataFrame(rows).to_string(index=False))
    print()


def evaluate_ensemble(
    meta_model, test_probs_list: list, y_test: np.ndarray, le_target
):
    """
    スタッキングメタ学習器を評価する。
    精度に加え Log Loss と多クラス Brier Score で確率較正の品質も測定する。
    """
    from sklearn.metrics import log_loss

    meta_X_test = np.hstack(test_probs_list)
    y_pred = meta_model.predict(meta_X_test)
    y_proba = meta_model.predict_proba(meta_X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    # 多クラス Brier Score: (1/n) * Σ_i Σ_c (y_ic - p_ic)^2
    n_classes = y_proba.shape[1]
    y_one_hot = np.eye(n_classes)[y_test]
    brier = float(np.mean(np.sum((y_one_hot - y_proba) ** 2, axis=1)))

    print(f"\n=== [Ensemble Stack] テスト精度: {acc:.4f} ({acc*100:.2f}%) ===")
    print(f"    Log Loss: {ll:.4f}  |  Brier Score: {brier:.4f}")
    print()

    labels = le_target.classes_
    names = [PITCH_LABEL_MAP.get(lb, lb) for lb in labels]
    print(classification_report(y_test, y_pred, target_names=names))

    return acc, y_pred


def evaluate_hybrid(model, ds_test, cfg_hybrid: dict, le_target, device):
    """Hybrid LSTMモデルのテスト精度を評価する。"""
    import torch
    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        ds_test, batch_size=cfg_hybrid["batch_size"],
        shuffle=False, num_workers=0,
    )

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for seq, tab, target, lengths in test_loader:
            seq, tab = seq.to(device), tab.to(device)
            pred = model(seq, lengths, tab).argmax(dim=1).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(target.numpy())

    y_pred = np.concatenate(all_preds)
    y_test = np.concatenate(all_targets)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== [Hybrid LSTM] テスト精度: {acc:.4f} ({acc*100:.2f}%) ===\n")

    labels = le_target.classes_
    names = [PITCH_LABEL_MAP.get(lb, lb) for lb in labels]
    print(classification_report(y_test, y_pred, target_names=names))

    return acc, y_pred, y_test
