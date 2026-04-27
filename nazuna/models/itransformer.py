from nazuna.models._base import BasicBaseModel
from nazuna.models.common import RevIN
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
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        d_model: int = 128, n_heads: int = 4, d_ff: int = 256,
        e_layers: int = 3, dropout: float = 0.1,
        revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
        scaler_cls: type | None = None,
        scaler_params: dict | None = None,
        prep_type: str = 'none',
    ) -> None:
        super()._setup(seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type)

        self.use_revin = revin
        if self.use_revin:
            self.revin = RevIN(c_in, affine=revin_affine, eps=revin_eps)

        self.embed = torch.nn.Linear(seq_len, d_model)
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
        if self.use_revin:
            x, x_mean, x_std = self.revin.normalize(x)
        # Invert: treat each variate as a token.
        h = x.transpose(1, 2)  # [B, C, L]
        h = self.embed(h)  # [B, C, d_model]
        h = self.encoder(h)  # [B, C, d_model]
        yhat = self.head(h)  # [B, C, pred_len]
        yhat = yhat.transpose(1, 2)  # [B, pred_len, C]
        if self.use_revin:
            yhat = self.revin.denormalize(yhat, x_mean, x_std)
        return yhat, {}
