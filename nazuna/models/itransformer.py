from nazuna.models._base import BasicBaseModel
from nazuna.models.common import \
    TimeFeatureEmbedding, MultiheadAttention, TransformerEncoderLayer
import numpy as np
import torch


class iTransformer(BasicBaseModel):
    """
    !!! info "Original Research"
        This model is based on the following research:
        > Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu,
          Shiyu Wang, Lintao Ma, and Mingsheng Long.
          "iTransformer: Inverted Transformers Are Effective
          for Time Series Forecasting."
          In Proceedings of the 12th International Conference on Learning
          Representations (ICLR 2024), 2024.
          [Paper](https://openreview.net/forum?id=JePfAI8fah) |
          [arXiv](https://arxiv.org/abs/2310.06625) |
          [GitHub](https://github.com/thuml/iTransformer)
    """
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        d_model: int = 512, n_heads: int = 8, d_ff: int = 512, e_layers: int = 2,
        dropout_emb: float = 0.1, dropout_aw: float = 0.1, dropout_sa: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.2), res_attention: bool = False,
        norm_first: bool = False, norm_out: bool = False, use_pos_enc: bool = False,
        use_time_features: bool = True, freq: str = 'hour',
        scaler_cls: type | None = None, scaler_params: dict | None = None,
        prep_type: str = 'none',
        use_revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
        use_lc: bool = False,
        lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        assert d_model % n_heads == 0, 'Expected d_model to be divisible by n_heads'
        assert d_model // n_heads >= 4, 'Expected head_dim >= 4'
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )

        self.use_time_features = use_time_features
        if self.use_time_features:
            self.tfe = TimeFeatureEmbedding(self.device, freq, d_model)
        self.embed = torch.nn.Linear(seq_len, d_model)
        self.dropout_emb = torch.nn.Dropout(dropout_emb)
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            # Learnable per-channel (variate) identity embedding.
            self.pos_enc = torch.nn.Parameter(torch.empty(c_in, d_model))
            torch.nn.init.uniform_(self.pos_enc, -0.02, 0.02)

        self.encoder_layers = torch.nn.ModuleList([
            TransformerEncoderLayer(
                MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw),
                d_model=d_model, d_ff=d_ff,
                norm_0=torch.nn.LayerNorm(d_model),
                norm_1=torch.nn.LayerNorm(d_model),
                activation=torch.nn.GELU(),
                dropout_sa=dropout_sa, dropout_ff=dropout_ff,
                norm_first=norm_first,
            )
            for _ in range(e_layers)
        ])
        self.res_attention = res_attention

        self.out_proj = torch.nn.Linear(d_model, pred_len)

        # Final normalization after the encoder stack, added last to keep the
        # random initialization of the other layers independent of norm_out.
        self.norm_out = torch.nn.LayerNorm(d_model) if norm_out else None

    def _extract_input(self, batch):
        x, prep_info = super()._extract_input(batch)
        x_mark = None
        if self.use_time_features:
            tsta = np.asarray(batch.tsta[:, -self.seq_len:])
            x_mark = self.tfe.get_feats(tsta)
        return (x, x_mark), prep_info

    def forward(self, input_):
        x, x_mark = input_  # x: (B, L, C), x: (B, L, n_feat)
        _, _, C = x.shape

        h = x.transpose(1, 2)  # (B, C, L)
        if x_mark is not None:
            h = torch.cat([h, x_mark.transpose(1, 2)], dim=1)  # (B, C + n_feat, L)
        h = self.embed(h)  # (B, C + n_feat, d_model)
        if self.use_pos_enc:
            # Add per-channel embedding to the data-channel tokens only.
            h = torch.cat([h[:, :C] + self.pos_enc.unsqueeze(0), h[:, C:]], dim=1)
        h = self.dropout_emb(h)

        scores = None
        for i, layer in enumerate(self.encoder_layers):
            h, scores, x_f1_debug = layer(h, (scores if self.res_attention else None))
            if i == len(self.encoder_layers) - 1:
                h_f1 = x_f1_debug[:, :C]  # (B, C, d_ff)
                f1_norm = torch.linalg.vector_norm(h_f1, dim=(1, 2)).mean().item()
                self._debug_if_initial_stage(f'x_f1_shape = {tuple(h_f1.shape)}')
                self._debug(f'x_f1_norm = {f1_norm}')
        if self.norm_out is not None:
            h = self.norm_out(h)

        h_out = h[:, :C]  # (B, C, d_model)
        out_norm = torch.linalg.vector_norm(h_out, dim=(1, 2)).mean().item()
        self._debug_if_initial_stage(f'x_out_shape = {tuple(h_out.shape)}')
        self._debug(f'x_out_norm = {out_norm}')
        self._finish_initial_debug_stage()

        y = self.out_proj(h)  # (B, C + n_feat, pred_len)
        y = y[:, :C, :]  # (B, C, pred_len)
        y = y.transpose(1, 2)  # (B, pred_len, C)
        return y, {}
