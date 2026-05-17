import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_PHYS_COLS = [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "balls",    # 打席内のカウント変化をLSTMに渡す
    "strikes",
]
SEQ_DIM = len(SEQ_PHYS_COLS) + 1  # +1: 球種エンコード値


class AtBatDataset(Dataset):
    """
    構築時に全シーケンスを最大長でゼロパディングしてテンソルで保持する。
    __getitem__ はテンソルスライスを返すだけなので、
    デフォルトの collate_fn で動作し、カスタム collate_fn は不要。

    seq   : (max_seq_len, SEQ_DIM)  打席内の直前投球シーケンス
    tab   : (tab_dim,)              現在の試合状況特徴量
    target: ()                      球種ラベル
    length: ()                      seq の実長（パディング前）
    """

    def __init__(self, df: pd.DataFrame, tab_cols: list, le_target, scalers: tuple):
        tab_mean, tab_std = scalers
        num_classes = len(le_target.classes_)
        # _pt_enc を [0,1] 範囲に正規化（1-9 の生整数をそのまま渡すとスケール不一致になる）
        pitch_enc = {p: float(i + 1) / num_classes for i, p in enumerate(le_target.classes_)}

        df = df.copy().reset_index(drop=True)
        df["_target"] = le_target.transform(df["pitch_type"].values)
        df["_pt_enc"] = df["pitch_type"].map(pitch_enc).fillna(0.0)

        # tab_cols で一括スケーリング（SEQ_PHYS_COLS は tab_cols に含まれるため2重スケーリングしない）
        df[tab_cols] = (df[tab_cols] - tab_mean) / tab_std
        df[tab_cols] = df[tab_cols].fillna(0.0)

        seq_cols_full = SEQ_PHYS_COLS + ["_pt_enc"]

        # 中間リストを作らず最初から preallocate した配列に直接書き込む。
        # 83K+ 個の小さな numpy 配列を積み上げて一括解放するとヒープが断片化し
        # DataLoader の torch.randperm でクラッシュする。
        max_seq_len = max(
            1,
            int(df.groupby(["game_pk", "at_bat_number"]).size().max()) - 1,
        )
        total_n = len(df)
        tab_dim = len(tab_cols)

        padded   = np.zeros((total_n, max_seq_len, SEQ_DIM), dtype=np.float32)
        tabs_arr = np.zeros((total_n, tab_dim),               dtype=np.float32)
        tgts_arr = np.zeros(total_n,                          dtype=np.int64)
        lens_arr = np.zeros(total_n,                          dtype=np.int64)   # 0 = 直前投球なし（初球）

        idx = 0
        for (_, _), group in df.groupby(["game_pk", "at_bat_number"], sort=False):
            group   = group.sort_values("pitch_number").reset_index(drop=True)
            seq_all = group[seq_cols_full].fillna(0.0).values.astype(np.float32)
            tab_all = group[tab_cols].values.astype(np.float32)
            tgt_all = group["_target"].values.astype(np.int64)

            for i in range(len(group)):
                if i > 0:
                    padded[idx, :i] = seq_all[:i]   # view をコピー（小オブジェクト不要）
                    lens_arr[idx] = i
                tabs_arr[idx] = tab_all[i]
                tgts_arr[idx] = int(tgt_all[i])
                idx += 1

        self.seqs    = torch.from_numpy(padded)
        self.tabs    = torch.from_numpy(tabs_arr)
        self.targets = torch.from_numpy(tgts_arr)
        self.lengths = torch.from_numpy(lens_arr)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        return self.seqs[idx], self.tabs[idx], self.targets[idx], self.lengths[idx]


def fit_scalers(df_train: pd.DataFrame, tab_cols: list) -> tuple:
    tab_mean = df_train[tab_cols].mean()
    tab_std = df_train[tab_cols].std().replace(0, 1)
    return tab_mean, tab_std
