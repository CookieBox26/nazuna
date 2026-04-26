from nazuna.models._base import BasicBaseModel
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
    """
    def _get_seq_len_for_model(self, seq_len):
        return seq_len

    def _setup(
        self, seq_len: int, pred_len: int,
        d_model: int = 128, n_heads: int = 4, d_ff: int = 256,
        e_layers: int = 3, dropout: float = 0.1,
        scaler_cls: type | None = None,
        scaler_params: dict | None = None,
    ) -> None:
        super()._setup(seq_len, pred_len, scaler_cls, scaler_params)
        seq_len_for_model = self._get_seq_len_for_model(seq_len)

        self.embed = torch.nn.Linear(seq_len_for_model, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=False,
            activation='gelu',
        )
        self.encoder = torch.nn.TransformerEncoder(
            enc_layer, num_layers=e_layers, enable_nested_tensor=False,
        )
        self.head = torch.nn.Linear(d_model, pred_len)

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
    def _get_seq_len_for_model(self, seq_len):
        return seq_len - 1

    def forward(self, x):  # x: [B, seq_len, C] (scaled)
        last_val = x[:, -1:, :]  # [B, 1, C]
        dx = x[:, 1:, :] - x[:, :-1, :]  # [B, seq_len-1, C]
        pred_dx, info = super().forward(dx)  # [B, pred_len, C]
        pred = last_val + torch.cumsum(pred_dx, dim=1)
        return pred, info
