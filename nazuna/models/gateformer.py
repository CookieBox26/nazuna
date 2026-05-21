from nazuna.models._base import BasicBaseModel
from nazuna.models.common import \
    MultiheadAttention, TransformerEncoderLayer
import math
import torch
import torch.nn.functional as F


class Gateformer(BasicBaseModel):
    """
    !!! info "Original Research"
        This model is based on the following research:
        > Yu-Hsiang Chen, Hsiao-Hua Chang, Chia-Wen Chen, Si-An Chen,
          Hsiang-Fu Yu, and Cho-Jui Hsieh.
          "Gateformer: Advancing Multivariate Time Series Forecasting through
          Temporal and Variate-Wise Attention with Gated Representations."
          arXiv preprint, 2025.
          [arXiv](https://arxiv.org/abs/2505.00307) |
          [GitHub](https://github.com/nyuolab/Gateformer)
    """
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        patch_len: int = 8, stride: int = 8,
        d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, e_layers: int = 2,
        dropout_emb: float = 0.1, dropout_aw: float = 0.1, dropout_sa: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.2),
        scaler_cls: type | None = None, scaler_params: dict | None = None,
        prep_type: str = 'none',
        use_revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
        use_lc: bool = False,
        lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        assert seq_len >= patch_len, 'seq_len >= patch_len'
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )

        self.patch_len = patch_len
        self.stride = stride
        # Replication padding by `stride` on the right; mirrors original Gateformer.
        self.padding = stride
        self.n_patches = (seq_len + self.padding - patch_len) // stride + 1

        self.patch_proj = torch.nn.Linear(patch_len, d_model, bias=False)
        pe = self._build_sinusoidal_pe(self.n_patches, d_model)
        self.register_buffer('pos_enc', pe)
        self.dropout_emb = torch.nn.Dropout(dropout_emb)

        self.global_proj = torch.nn.Linear(seq_len, d_model)
        self.dropout_global = torch.nn.Dropout(dropout_emb)

        self.enc_temporal = torch.nn.ModuleList([
            self._build_encoder_layer(
                d_model, n_heads, d_ff, dropout_aw, dropout_sa, dropout_ff,
            )
            for _ in range(e_layers)
        ])
        self.enc_variate = torch.nn.ModuleList([
            self._build_encoder_layer(
                d_model, n_heads, d_ff, dropout_aw, dropout_sa, dropout_ff,
            )
            for _ in range(e_layers)
        ])

        head_nf = d_model * self.n_patches
        self.head_flatten = torch.nn.Flatten(start_dim=-2)
        self.head_linear = torch.nn.Linear(head_nf, d_model)
        self.head_dropout = torch.nn.Dropout(dropout_emb)

        self.gate_w1 = torch.nn.Linear(d_model, d_model)
        self.gate_w2 = torch.nn.Linear(d_model, d_model)
        self.gate_w3 = torch.nn.Linear(d_model, d_model)
        self.gate_w4 = torch.nn.Linear(d_model, d_model)

        self.out_proj = torch.nn.Linear(d_model, pred_len, bias=True)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    @staticmethod
    def _build_encoder_layer(
        d_model, n_heads, d_ff, dropout_aw, dropout_sa, dropout_ff,
    ):
        return TransformerEncoderLayer(
            MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw),
            d_model=d_model, d_ff=d_ff,
            norm_0=torch.nn.LayerNorm(d_model, eps=1e-5),
            norm_1=torch.nn.LayerNorm(d_model, eps=1e-5),
            activation=torch.nn.ReLU(),
            dropout_sa=dropout_sa, dropout_ff=dropout_ff,
            norm_first=True,
        )

    def _patchify(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = F.pad(x, (0, self.padding), mode='replicate')
        # (B, C, L + padding) -> (B, C, P, patch_len)
        return x.unfold(dimension=2, size=self.patch_len, step=self.stride)

    def forward(self, x):
        B, L, C = x.shape

        global_h = self.global_proj(x.transpose(1, 2))  # (B, C, d_model)
        global_h = self.dropout_global(global_h)

        patches = self._patchify(x)  # (B, C, P, patch_len)
        P = patches.size(2)
        z = patches.reshape(B * C, P, self.patch_len)
        z = self.patch_proj(z) + self.pos_enc  # (B*C, P, d_model)
        z = self.dropout_emb(z)

        for layer in self.enc_temporal:
            z, _ = layer(z, None)

        z = z.reshape(B, C, P, -1)  # (B, C, P, d_model)
        z = z.transpose(2, 3)  # (B, C, d_model, P)
        temporal_h = self.head_flatten(z)  # (B, C, d_model * P)
        temporal_h = self.head_linear(temporal_h)  # (B, C, d_model)
        temporal_h = self.head_dropout(temporal_h)

        gate = torch.sigmoid(self.gate_w1(global_h) + self.gate_w2(temporal_h))
        h = gate * global_h + (1.0 - gate) * temporal_h

        h_cross = h
        for layer in self.enc_variate:
            h_cross, _ = layer(h_cross, None)

        gate = torch.sigmoid(self.gate_w3(h) + self.gate_w4(h_cross))
        h = gate * h + (1.0 - gate) * h_cross

        y = self.out_proj(h)  # (B, C, pred_len)
        y = y.transpose(1, 2)  # (B, pred_len, C)
        return y, {}
