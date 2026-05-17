import torch
import torch.nn as nn


class HybridLSTMClassifier(nn.Module):
    """
    打席内の投球シーケンス(LSTM+Attention) + 試合状況特徴量(全結合) のハイブリッドモデル。

    LSTM branch:
        直前投球の物理量・カウントシーケンス → Self-Attention で重み付け平均 h
    Tabular branch:
        カウント・ランナー・球速・変化量など現在の特徴量
    Head:
        [h, tab] を結合 → BatchNorm付き全結合層 → 球種予測

    Note: pack_padded_sequence は MPS で segfault するため使わない。
          Self-Attention でパディング位置をマスクして有効タイムステップのみ集約する。
    """

    def __init__(
        self,
        seq_dim: int,
        tab_dim: int,
        lstm_hidden: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 打席先頭を表す学習可能な BOS トークン。
        # 初球（直前投球なし）でもゼロ列ではなく意味のある「打席開始」表現を LSTM に渡す。
        self.bos = nn.Parameter(torch.zeros(1, 1, seq_dim))
        # 各タイムステップの重みを学習するアテンションスコアラー
        self.attn = nn.Linear(lstm_hidden, 1, bias=False)

        combined_dim = lstm_hidden + tab_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        # XGBoostが物理量→球種を直接学習できるのと同等の経路を持たせるSkip Connection。
        # 特に初球など履歴が少ない場面でLSTM枝の補完として機能する。
        self.tab_skip = nn.Linear(tab_dim, num_classes)

    def forward(
        self,
        seq: torch.Tensor,      # (batch, max_seq_len, seq_dim)
        lengths: torch.Tensor,   # (batch,) — パディング前の実長
        tab: torch.Tensor,      # (batch, tab_dim)
    ) -> torch.Tensor:
        # BOS トークンを先頭に付加: 初球はBOSのみ、N球目はBOS+N-1投球
        bos = self.bos.expand(seq.size(0), -1, -1)
        seq = torch.cat([bos, seq], dim=1)  # (batch, max_len+1, seq_dim)

        output, _ = self.lstm(seq)  # (batch, max_len+1, lstm_hidden)

        # BOS は常に有効、その後に lengths 個の実投球が続く
        lengths_with_bos = (lengths + 1).to(output.device)
        max_len = output.size(1)
        mask = torch.arange(max_len, device=output.device).unsqueeze(0) < lengths_with_bos.unsqueeze(1)

        attn_scores = self.attn(output).squeeze(-1)          # (batch, max_len)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)     # (batch, max_len)
        h = (output * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, lstm_hidden)

        # 初球（lengths=0）はLSTM出力をゼロにして head が tab のみで予測。
        # 直前投球の文脈がない状態で BOS 由来の h をそのまま使うとノイズになる。
        # head 内の非線形変換（BN+ReLU）が XGBoost と同様に物理量から球種を識別する。
        has_history = (lengths > 0).float().unsqueeze(1).to(h.device)
        h = h * has_history

        x = torch.cat([h, tab], dim=1)
        return self.head(x) + self.tab_skip(tab)
