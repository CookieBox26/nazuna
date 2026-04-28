from nazuna.models._base import BasicBaseModel
from nazuna.models.common import RevIN, TimeFeatureEmbedding
import numpy as np
import torch


class iTransformer(BasicBaseModel):
    """
    !!! note "Original Research"
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

    !!! tip "Standard parameter settings"
        ```toml
        [definitions.iTransformer]
        cls_path = "nazuna.models.itransformer.iTransformer"
        [definitions.iTransformer.params]
        seq_len = 96  # task-dependent
        pred_len = 24  # task-dependent
        c_in = 10  # task-dependent
        d_model = 128
        n_heads = 4
        d_ff = 256
        e_layers = 3
        dropout = 0.1
        revin = true
        revin_affine = false
        revin_eps = 1e-5
        use_time_features = true
        freq = "hour"
        norm = true
        scaler_cls_path = ""
        scaler_params = {}
        prep_type = "none"
        ```
    """
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        d_model: int = 128, n_heads: int = 4, d_ff: int = 256,
        e_layers: int = 3, dropout: float = 0.1,
        revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
        use_time_features: bool = True, freq: str = 'hour', norm: bool = True,
        scaler_cls: type | None = None, scaler_params: dict | None = None,
        prep_type: str = 'none',
    ) -> None:
        super()._setup(seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type)
        self.c_in = c_in

        self.use_revin = revin
        if self.use_revin:
            self.revin = RevIN(c_in, affine=revin_affine, eps=revin_eps)

        self.use_time_features = use_time_features
        if self.use_time_features:
            self.tfe = TimeFeatureEmbedding(self.device, freq, d_model)

        self.embed = torch.nn.Linear(seq_len, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=False, activation='gelu',
        )
        self.encoder = torch.nn.TransformerEncoder(
            enc_layer, num_layers=e_layers, enable_nested_tensor=False,
        )
        self.use_norm = norm
        if self.use_norm:
            self.enc_norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, pred_len)

    def _extract_input(self, batch):
        x, current_value = super()._extract_input(batch)
        x_mark = None
        if self.use_time_features:
            tsta = np.asarray(batch.tsta[:, -self.seq_len:])
            x_mark = self.tfe.get_feats(tsta)
        return (x, x_mark), current_value

    def forward(self, input_):
        x, x_mark = input_  # x: [B, L, C], x: [B, L, n_feat]
        if self.use_revin:
            x, x_mean, x_std = self.revin.normalize(x)
        # Invert: treat each variate as a token.
        h = x.transpose(1, 2)  # [B, C, L]
        if x_mark is not None:
            # Append time-feature tokens along the variate axis.
            h = torch.cat([h, x_mark.transpose(1, 2)], dim=1)  # [B, C + n_feat, L]
        h = self.embed(h)  # [B, C (+ n_feat), d_model]
        h = self.encoder(h)
        if self.use_norm:
            h = self.enc_norm(h)
        yhat = self.head(h)  # [B, C (+ n_feat), pred_len]
        yhat = yhat[:, :self.c_in, :]  # drop time-feature tokens
        yhat = yhat.transpose(1, 2)  # [B, pred_len, C]
        if self.use_revin:
            yhat = self.revin.denormalize(yhat, x_mean, x_std)
        return yhat, {}
