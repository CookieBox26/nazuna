from nazuna.models._base import BasicBaseModel
from nazuna.models.common import \
    MultiheadAttention, TransformerEncoderLayer, Patchifier, TimeFeatureEmbedding
import math
import numpy as np
import torch


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
        patch_len: int = 8, stride: int = 8, padding_patch: str | None = 'end',
        d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, e_layers: int = 2,
        dropout_emb: float = 0.1, dropout_aw: float = 0.1, dropout_sa: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.2), res_attention: bool = False,
        norm_first: bool = True,
        d_model_variate: int = -1, n_heads_variate: int = -1, d_ff_variate: int = -1,
        e_layers_variate: int = -1, res_attention_variate: bool | None = None,
        norm_first_variate: bool | None = None,
        use_time_features: bool = False, freq: str = 'hour',
        scaler_cls: type | None = None, scaler_params: dict | None = None,
        prep_type: str = 'none',
        use_revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
        use_lc: bool = False,
        lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        assert seq_len >= patch_len >= stride, 'Expected seq_len >= patch_len >= stride'
        assert d_model % n_heads == 0, 'Expected d_model to be divisible by n_heads'
        assert d_model // n_heads >= 4, 'Expected head_dim >= 4'
        d_model_variate = d_model if d_model_variate == -1 else d_model_variate
        n_heads_variate = n_heads if n_heads_variate == -1 else n_heads_variate
        d_ff_variate = d_ff if d_ff_variate == -1 else d_ff_variate
        e_layers_variate = e_layers if e_layers_variate == -1 else e_layers_variate
        res_attention_variate = (
            res_attention if res_attention_variate is None else res_attention_variate
        )
        norm_first_variate = (
            norm_first if norm_first_variate is None else norm_first_variate
        )
        assert d_model_variate % n_heads_variate == 0, \
            'Expected d_model_variate to be divisible by n_heads_variate'
        assert d_model_variate // n_heads_variate >= 4, \
            'Expected variate head_dim >= 4'
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )

        self.use_time_features = use_time_features
        if self.use_time_features:
            self.tfe = TimeFeatureEmbedding(self.device, freq, d_model)

        self.patch_len = patch_len
        self.patchifier = Patchifier(patch_len, stride, padding_patch)
        self.n_patches = self.patchifier.num_patches(seq_len)

        self.patch_proj = torch.nn.Linear(patch_len, d_model, bias=False)
        pe = self._build_sinusoidal_pe(self.n_patches, d_model)
        self.register_buffer('pos_enc', pe)
        self.dropout_emb = torch.nn.Dropout(dropout_emb)

        self.global_proj = torch.nn.Linear(seq_len, d_model_variate)
        self.dropout_global = torch.nn.Dropout(dropout_emb)

        self.enc_temporal = torch.nn.ModuleList([
            self._build_encoder_layer(
                d_model, n_heads, d_ff, dropout_aw, dropout_sa, dropout_ff,
                norm_first,
            )
            for _ in range(e_layers)
        ])
        self.enc_variate = torch.nn.ModuleList([
            self._build_encoder_layer(
                d_model_variate, n_heads_variate, d_ff_variate,
                dropout_aw, dropout_sa, dropout_ff, norm_first_variate,
            )
            for _ in range(e_layers_variate)
        ])
        self.res_attention = res_attention
        self.res_attention_variate = res_attention_variate

        head_nf = d_model * self.n_patches
        self.head_flatten = torch.nn.Flatten(start_dim=-2)
        self.head_linear = torch.nn.Linear(head_nf, d_model_variate)
        self.head_dropout = torch.nn.Dropout(dropout_emb)

        self.gate_w1 = torch.nn.Linear(d_model_variate, d_model_variate)
        self.gate_w2 = torch.nn.Linear(d_model_variate, d_model_variate)
        self.gate_w3 = torch.nn.Linear(d_model_variate, d_model_variate)
        self.gate_w4 = torch.nn.Linear(d_model_variate, d_model_variate)

        self.out_proj = torch.nn.Linear(d_model_variate, pred_len, bias=True)

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
        d_model, n_heads, d_ff, dropout_aw, dropout_sa, dropout_ff, norm_first,
    ):
        return TransformerEncoderLayer(
            MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw),
            d_model=d_model, d_ff=d_ff,
            norm_0=torch.nn.LayerNorm(d_model, eps=1e-5),
            norm_1=torch.nn.LayerNorm(d_model, eps=1e-5),
            activation=torch.nn.ReLU(),
            dropout_sa=dropout_sa, dropout_ff=dropout_ff,
            norm_first=norm_first,
        )

    def _extract_input(self, batch):
        x, prep_info = super()._extract_input(batch)
        x_mark = None
        if self.use_time_features:
            tsta = np.asarray(batch.tsta[:, -self.seq_len:])
            x_mark = self.tfe.get_feats(tsta)
        return (x, x_mark), prep_info

    def forward(self, input_):
        x, x_mark = input_  # x: (B, L, C), x_mark: (B, L, n_feat) or None
        B, L, C = x.shape

        global_h = self.global_proj(x.transpose(1, 2))  # (B, C, d_model_variate)
        global_h = self.dropout_global(global_h)

        patches = self.patchifier(x)  # (B, C, P, patch_len)
        P = patches.size(2)
        z = patches.reshape(B * C, P, self.patch_len)
        z = self.patch_proj(z) + self.pos_enc  # (B*C, P, d_model)
        z = self.dropout_emb(z)

        scores = None
        for layer in self.enc_temporal:
            z, scores = layer(z, (scores if self.res_attention else None))

        z = z.reshape(B, C, P, -1)  # (B, C, P, d_model)
        z = z.transpose(2, 3)  # (B, C, d_model, P)
        temporal_h = self.head_flatten(z)  # (B, C, d_model * P)
        temporal_h = self.head_linear(temporal_h)  # (B, C, d_model_variate)
        temporal_h = self.head_dropout(temporal_h)

        gate = torch.sigmoid(self.gate_w1(global_h) + self.gate_w2(temporal_h))
        h = gate * global_h + (1.0 - gate) * temporal_h

        if x_mark is not None:
            tf = self.global_proj(x_mark.transpose(1, 2))  # (B, n_feat, d_model_variate)
            tf = self.dropout_global(tf)
            h = torch.cat([h, tf], dim=1)  # (B, C + n_feat, d_model_variate)

        h_cross = h
        scores = None
        for layer in self.enc_variate:
            h_cross, scores = layer(
                h_cross, (scores if self.res_attention_variate else None),
            )

        gate = torch.sigmoid(self.gate_w3(h) + self.gate_w4(h_cross))
        h = gate * h + (1.0 - gate) * h_cross

        y = self.out_proj(h)  # (B, C + n_feat, pred_len)
        y = y[:, :C, :]  # (B, C, pred_len)
        y = y.transpose(1, 2)  # (B, pred_len, C)
        return y, {}
