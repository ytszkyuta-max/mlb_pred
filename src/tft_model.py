import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """
    Gated Residual Network (Lim et al. 2021 §2.2).

    GRN(a, c) = LayerNorm(skip(a) + GLU(W₁·ELU(W₂·a + W₃·c)))
    GLU(x)    = σ(W₄x) ⊙ W₅x

    HybridLSTM の全結合層と異なり、GLU ゲーティングで
    不要な情報を閾値的にカットできる。
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int | None = None,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ):
        super().__init__()
        d_out = d_out or d_in
        self.context_proj = nn.Linear(context_dim, d_hidden, bias=False) if context_dim else None
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate1 = nn.Linear(d_out, d_out)   # σ branch of GLU
        self.gate2 = nn.Linear(d_out, d_out)   # linear branch of GLU
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if context is not None and self.context_proj is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.dropout(h)
        h1 = self.fc2(h)
        glu = torch.sigmoid(self.gate1(h1)) * self.gate2(h1)
        return self.norm(self.skip(x) + glu)


class VariableSelectionNetwork(nn.Module):
    """
    VSN: 各スカラー変数を個別 GRN で d_model に変換し、
    softmax 重みで加重和して選択的に集約する。

    「球速が重要か回転数が重要か」をカウント状況ごとに動的に学習できる。
    """

    def __init__(
        self,
        num_vars: int,
        d_model: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.var_grns = nn.ModuleList([GRN(1, d_model, d_model, dropout) for _ in range(num_vars)])
        self.weight_grn = GRN(num_vars, d_model, num_vars, dropout, context_dim=context_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        x       : (batch, num_vars) or (batch, seq_len, num_vars)
        context : (batch, context_dim)
        returns : (batch, d_model) or (batch, seq_len, d_model)
        """
        is_seq = x.dim() == 3
        if is_seq:
            B, T, V = x.shape
            x_flat = x.reshape(B * T, V)
            ctx_flat = context.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1) if context is not None else None
        else:
            x_flat = x
            ctx_flat = context

        var_outs = torch.stack(
            [grn(x_flat[:, i:i+1]) for i, grn in enumerate(self.var_grns)], dim=1
        )  # (batch_flat, num_vars, d_model)

        weights = torch.softmax(self.weight_grn(x_flat, ctx_flat), dim=-1)  # (batch_flat, num_vars)
        out = (var_outs * weights.unsqueeze(-1)).sum(dim=1)                   # (batch_flat, d_model)

        return out.reshape(B, T, self.d_model) if is_seq else out


class TFTPitchClassifier(nn.Module):
    """
    Temporal Fusion Transformer for MLB pitch type classification.

    HybridLSTMClassifier との主な違い:
    - GRN: ELU + GLU ゲーティングで学習可能な非線形フィルタリング
    - VSN: 投球物理量 (球速・回転数・変化量) の各変数に重要度スコアを付けて選択的集約
    - 静的コンテキスト: tab (投手・打者特徴) で LSTM 初期状態とエンリッチメントを条件付け
    - Multi-head self-attention: 打席内の長距離依存関係を複数の視点で捕捉

    Interface: forward(seq, lengths, tab) → logits  （HybridLSTMClassifier と互換）
    """

    def __init__(
        self,
        seq_dim: int,
        tab_dim: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_lstm_layers = num_lstm_layers

        # ── Static branch (tab 特徴量) ────────────────────────────────────────
        # tab 全体を GRN で d_model に集約し、4 種のコンテキストベクトルを派生
        self.static_grn = GRN(tab_dim, d_model * 2, d_model, dropout)
        self.ctx_h = GRN(d_model, d_model, dropout=dropout)   # LSTM h₀ 初期状態
        self.ctx_c = GRN(d_model, d_model, dropout=dropout)   # LSTM c₀ 初期状態
        self.ctx_e = GRN(d_model, d_model, dropout=dropout)   # エンリッチメント
        self.ctx_s = GRN(d_model, d_model, dropout=dropout)   # seq VSN への文脈

        # ── Temporal branch (seq 特徴量) ──────────────────────────────────────
        # 打席開始を表す学習可能な BOS トークン
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        # VSN で各投球の物理量特徴を変換 (静的コンテキストで条件付け)
        self.seq_vsn = VariableSelectionNetwork(seq_dim, d_model, dropout, context_dim=d_model)

        # LSTM: 静的コンテキストで初期状態を条件付け
        self.lstm = nn.LSTM(
            d_model, d_model, num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # ── Enrichment & Attention ────────────────────────────────────────────
        # LSTM 出力に静的文脈を付加して情報を豊かにする
        self.enrichment_grn = GRN(d_model, d_model * 2, d_model, dropout, context_dim=d_model)
        # 打席内投球間の依存関係を複数ヘッドで捕捉
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_grn = GRN(d_model, d_model * 2, d_model, dropout)   # post-attention gate
        self.ff_grn = GRN(d_model, d_model * 4, d_model, dropout)     # position-wise FF
        self.output_norm = nn.LayerNorm(d_model)

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Linear(d_model, num_classes)
        # 静的コンテキストからの直接経路: 初球など投球履歴がない場面のスキップ接続
        self.static_skip = nn.Linear(d_model, num_classes)

    def forward(
        self,
        seq: torch.Tensor,       # (batch, max_seq_len, seq_dim)
        lengths: torch.Tensor,    # (batch,) — パディング前の実投球数
        tab: torch.Tensor,        # (batch, tab_dim)
    ) -> torch.Tensor:

        B = seq.size(0)
        device = seq.device

        # ── 1. Static feature processing ──────────────────────────────────────
        static_enc = self.static_grn(tab)                    # (batch, d_model)
        h0_ctx = self.ctx_h(static_enc)
        c0_ctx = self.ctx_c(static_enc)
        enr_ctx = self.ctx_e(static_enc)
        seq_ctx = self.ctx_s(static_enc)

        # ── 2. Temporal feature processing via VSN ────────────────────────────
        seq_enc = self.seq_vsn(seq, seq_ctx)                 # (batch, max_seq_len, d_model)

        bos = self.bos.expand(B, -1, -1)
        seq_with_bos = torch.cat([bos, seq_enc], dim=1)      # (batch, max_len+1, d_model)
        max_len = seq_with_bos.size(1)

        # ── 3. LSTM with static context as initial state ──────────────────────
        h0 = h0_ctx.unsqueeze(0).expand(self.num_lstm_layers, -1, -1).contiguous()
        c0 = c0_ctx.unsqueeze(0).expand(self.num_lstm_layers, -1, -1).contiguous()
        lstm_out, _ = self.lstm(seq_with_bos, (h0, c0))      # (batch, max_len+1, d_model)

        # ── 4. Static enrichment ──────────────────────────────────────────────
        enr_exp = enr_ctx.unsqueeze(1).expand(-1, max_len, -1)
        enriched = self.enrichment_grn(lstm_out, enr_exp)    # (batch, max_len+1, d_model)

        # ── 5. Multi-head self-attention with key padding mask ────────────────
        lengths_with_bos = (lengths + 1).to(device)
        # True = パディング位置（無効）
        key_mask = (
            torch.arange(max_len, device=device).unsqueeze(0)
            >= lengths_with_bos.unsqueeze(1)
        )

        attn_out, _ = self.self_attn(
            enriched, enriched, enriched,
            key_padding_mask=key_mask,
            need_weights=False,
        )

        gated = self.attn_grn(attn_out + enriched)
        ff_out = self.ff_grn(gated)
        out = self.output_norm(ff_out + gated)               # (batch, max_len+1, d_model)

        # ── 6. 最終有効位置を取得して分類 ────────────────────────────────────
        # BOS=0, 1投目=1, ..., k投目=k なので last_idx = lengths
        last_idx = lengths.to(device)
        repr_vec = out[torch.arange(B, device=device), last_idx]   # (batch, d_model)

        return self.classifier(repr_vec) + self.static_skip(static_enc)
