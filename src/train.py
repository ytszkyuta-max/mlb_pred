import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import pandas as pd


def time_split(df, feature_cols: list, y: np.ndarray, test_ratio: float, val_ratio: float):
    """
    game_date 基準の時系列分割。
    同一試合・同一投手の投球が train/test に混在するデータリークを防ぐ。
    df は事前に game_date 昇順でソートされている前提。
    """
    n = len(df)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    train_n = n - test_n - val_n

    X = df[feature_cols].astype(float).values

    X_train, y_train = X[:train_n], y[:train_n]
    X_val, y_val = X[train_n:train_n + val_n], y[train_n:train_n + val_n]
    X_test, y_test = X[train_n + val_n:], y[train_n + val_n:]
    df_test = df.iloc[train_n + val_n:].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, df_test


def train(X_train, y_train, X_val, y_val, cfg: dict):
    # クラス不均衡対策: KC(1.8%), FS(3.0%) などの希少球種の学習を補正
    sample_weight = compute_sample_weight("balanced", y_train)

    model = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        objective="multi:softmax",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


# ── CatBoost ───────────────────────────────────────────────────────────────────

def train_catboost(X_train, y_train, X_val, y_val, cfg: dict, cat_feature_indices: list):
    """
    CatBoostClassifier を訓練する。
    カテゴリ特徴量 (stand, p_throws, prev_pitch_type, pitcher_enc) を文字列に変換して
    CatBoost の Pool へ渡す。CatBoost は float 配列に cat_features を受け付けないため、
    object 配列のカテゴリ列のみ str 化して対応する。
    クラス不均衡は auto_class_weights="Balanced" で自動補正。
    """
    from catboost import CatBoostClassifier, Pool

    def _to_pool(X, y):
        # object 配列に変換し、カテゴリ列だけ str 化（他列は float のまま保持）
        X_obj = X.astype(object)
        for idx in cat_feature_indices:
            X_obj[:, idx] = X[:, idx].astype(int).astype(str)
        return Pool(X_obj, y, cat_features=cat_feature_indices)

    train_pool = _to_pool(X_train, y_train)
    val_pool = _to_pool(X_val, y_val)

    model = CatBoostClassifier(
        iterations=cfg["iterations"],
        depth=cfg["depth"],
        learning_rate=cfg["learning_rate"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        loss_function="MultiClass",
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=50,
    )
    model.fit(train_pool, eval_set=val_pool)
    return model


def catboost_proba(cat_model, X: np.ndarray, cat_feature_indices: list) -> np.ndarray:
    """CatBoost の predict_proba をラップし、float 配列のカテゴリ列を str 化して Pool 経由で渡す。"""
    from catboost import Pool

    X_obj = X.astype(object)
    for idx in cat_feature_indices:
        X_obj[:, idx] = X[:, idx].astype(int).astype(str)
    pool = Pool(X_obj, cat_features=cat_feature_indices)
    return cat_model.predict_proba(pool)


def get_probs_hybrid(model, ds, cfg_hybrid: dict, device) -> np.ndarray:
    """HybridLSTM の softmax 確率行列 (n_samples, n_classes) を返す。"""
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds, batch_size=cfg_hybrid["batch_size"], shuffle=False, num_workers=0
    )
    probs_list = []
    model.eval()
    with torch.no_grad():
        for seq, tab, _, lengths in loader:
            seq, tab = seq.to(device), tab.to(device)
            logits = model(seq, lengths, tab)
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs_list)


def train_stacking(val_probs_list: list, y_val: np.ndarray, meta_C: float = 0.5):
    """
    ベースモデルの val 確率を横結合してメタ学習器 (LogisticRegression) を訓練する。
    入力: [(n_val, n_classes), ...] × n_base_models → (n_val, n_classes * n_base_models)
    """
    from sklearn.linear_model import LogisticRegression

    meta_X = np.hstack(val_probs_list)
    meta = LogisticRegression(
        max_iter=2000, random_state=42, C=meta_C, solver="lbfgs"
    )
    meta.fit(meta_X, y_val)
    return meta


class TemperatureScaler:
    """
    Post-hoc probability calibration via temperature scaling (Guo et al. 2017).
    Fits a single scalar T to minimize log loss on val set.
    softmax(log(p) / T): T>1 softens (less confident), T<1 sharpens.
    Accuracy is unchanged; only probability estimates are corrected.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        from scipy.optimize import minimize_scalar
        from sklearn.metrics import log_loss

        def objective(T):
            return log_loss(y, self._scale(proba, T))

        result = minimize_scalar(objective, bounds=(0.05, 20.0), method="bounded")
        self.temperature = float(result.x)
        return self

    def _scale(self, proba: np.ndarray, T: float) -> np.ndarray:
        log_p = np.log(np.clip(proba, 1e-10, 1.0))
        shifted = log_p / T
        shifted -= shifted.max(axis=1, keepdims=True)
        exp_p = np.exp(shifted)
        return exp_p / exp_p.sum(axis=1, keepdims=True)

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return self._scale(proba, self.temperature)


def train_blend_weights(val_probs_list: list, y_val: np.ndarray, n_trials: int = 300) -> list[float]:
    """
    Optuna で各モデルの重みを最適化するブレンディング。
    LogisticRegressionスタッキングよりパラメータ数が少なく過学習しにくい。
    戻り値: 正規化済み重みリスト（len == len(val_probs_list)）
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[blend] optuna未インストール。均等重みを使用します。")
        n = len(val_probs_list)
        return [1.0 / n] * n

    from sklearn.metrics import accuracy_score

    def objective(trial):
        weights = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(len(val_probs_list))])
        if weights.sum() < 1e-9:
            return 0.0
        weights /= weights.sum()
        blend = sum(w * p for w, p in zip(weights, val_probs_list))
        return accuracy_score(y_val, blend.argmax(axis=1))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw = np.array([study.best_params[f"w{i}"] for i in range(len(val_probs_list))])
    raw /= raw.sum()
    return raw.tolist()


# ── Hybrid LSTM ────────────────────────────────────────────────────────────────

def time_split_dfs(df: pd.DataFrame, test_ratio: float, val_ratio: float):
    """XGBoost用 time_split と同じ基準でDataFrameとして返す。"""
    n = len(df)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    train_n = n - test_n - val_n
    return (
        df.iloc[:train_n].copy(),
        df.iloc[train_n:train_n + val_n].copy(),
        df.iloc[train_n + val_n:].copy(),
    )


def train_hybrid(ds_train, ds_val, model, cfg_hybrid: dict, device):
    """
    HybridLSTMClassifier の学習ループ。
    val accuracyが最良のエポックのweightを最終モデルとして返す。
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Sampler

    class _NumpySampler(Sampler):
        """torch.randperm の代わりに numpy を使うシャッフルサンプラー。
        PyTorch 2.8.x の torch.randperm が大きな n でクラッシュするバグの回避策。"""
        def __init__(self, n: int) -> None:
            self.n = n
        def __iter__(self):
            return iter(np.random.permutation(self.n).tolist())
        def __len__(self) -> int:
            return self.n

    train_loader = DataLoader(
        ds_train, batch_size=cfg_hybrid["batch_size"],
        sampler=_NumpySampler(len(ds_train)), num_workers=0,
    )
    val_loader = DataLoader(
        ds_val, batch_size=cfg_hybrid["batch_size"],
        shuffle=False, num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_hybrid["lr"])

    # XGBoostと同様にクラス不均衡を補正（KC・FSなど希少球種の学習を強化）
    targets_np = ds_train.targets.numpy()
    classes = np.unique(targets_np)
    cw = compute_class_weight("balanced", classes=classes, y=targets_np)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(device),
    )

    # val精度が改善しなくなったらlrを半減（patience=5: 3では早期に下がりすぎる）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg_hybrid["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for seq, tab, target, lengths in train_loader:
            # lengths はpack_padded_sequenceのためCPUのまま渡す
            seq, tab, target = seq.to(device), tab.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq, lengths, tab), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seq, tab, target, lengths in val_loader:
                seq, tab, target = seq.to(device), tab.to(device), target.to(device)
                pred = model(seq, lengths, tab).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += len(target)

        val_acc = correct / total
        scheduler.step(1.0 - val_acc)
        print(
            f"Epoch {epoch:>3}/{cfg_hybrid['epochs']} "
            f"| loss={total_loss / len(train_loader):.4f} "
            f"| val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
