from nazuna.models.base import BasicBaseModel
from nazuna.scaler import IqrScaler
import torch
import torch.nn as nn


class iTransformer(BasicBaseModel):
    """
    !!! note "Original Research"
        This model is based on the following research:
        > Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu,
          Shiyu Wang, Lintao Ma, and Mingsheng Long.
          "iTransformer: Inverted Transformers Are Effective
          for Time Series Forecasting."
          In International Conference on Learning
          Representations (ICLR), 2024.
          [Paper](https://arxiv.org/abs/2310.06625) |
          [GitHub](https://github.com/thuml/iTransformer)
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        quantile_mode_train: str,
        quantile_mode_eval: str,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        e_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super()._setup(seq_len, pred_len)

        # Embed each variate's full time series into d_model.
        self.embed = nn.Linear(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=e_layers,
            enable_nested_tensor=False,
        )

        self.head = nn.Linear(d_model, pred_len)
        self.scaler = IqrScaler(
            quantile_mode_train, quantile_mode_eval
        )

    def forward(self, x):
        # x: [B, L, C]
        # Invert: treat each variate as a token.
        h = x.transpose(1, 2)  # [B, C, L]
        h = self.embed(h)  # [B, C, d_model]
        h = self.encoder(h)  # [B, C, d_model]
        yhat = self.head(h)  # [B, C, pred_len]
        yhat = yhat.transpose(1, 2)  # [B, pred_len, C]
        return yhat, {}


class DiffiTransformer(iTransformer):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        quantile_mode_train: str,
        quantile_mode_eval: str,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        e_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        # After first-order differencing, length becomes seq_len - 1.
        diff_seq_len = seq_len - 1
        super()._setup(
            diff_seq_len, pred_len,
            quantile_mode_train, quantile_mode_eval,
            d_model, n_heads, d_ff, e_layers, dropout,
        )
        # Restore original seq_len for _extract_input slicing.
        self.seq_len = seq_len

    def forward(self, x):  # x: [B, seq_len, C] (scaled)
        last_val = x[:, -1:, :]  # [B, 1, C]
        dx = x[:, 1:, :] - x[:, :-1, :]  # [B, seq_len-1, C]
        pred_dx, info = super().forward(dx)  # [B, pred_len, C]
        pred = last_val + torch.cumsum(pred_dx, dim=1)
        return pred, info
