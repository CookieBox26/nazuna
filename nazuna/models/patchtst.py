from nazuna.models.base import BasicBaseModel
from nazuna.scaler import IqrScaler
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):  # x: [B, S, D]
        s = x.size(1)
        return self.dropout(x + self.pe[:s].unsqueeze(0))


class PatchTST(BasicBaseModel):
    """
    !!! note "Original Research"
        This model is based on the following research:
        > Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam.
          "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers."
          In International Conference on Learning Representations (ICLR), 2023.
          [Paper](https://arxiv.org/abs/2211.14730) |
          [GitHub](https://github.com/yuqinie98/PatchTST)
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        quantile_mode: str,
    ) -> None:
        """
        Args:
            seq_len: Input sequence length (must be >= `patch_len`)
            pred_len: Prediction length
            quantile_mode: Source of quantiles for scaling ('full', 'cum', or 'rolling')

        Note:
            Fixed architecture parameters: `patch_len=8`, `stride=4`, `d_model=32`,
            `n_heads=2`, `n_layers=2`, `d_ff=128`, `dropout=0.1`
        """
        super()._setup(seq_len, pred_len)

        self.patch_len = 8
        self.stride = 4
        self.d_model = 32
        self.n_heads = 2
        self.n_layers = 2
        self.d_ff = 128
        self.dropout = 0.1
        self.use_layernorm_patch = True
        self.pool = 'last'  # 'last' or 'mean'

        assert seq_len >= self.patch_len, 'seq_len >= patch_len'
        self.n_patches = 1 + (self.seq_len - self.patch_len) // self.stride

        self.patch_proj = nn.Linear(self.patch_len, self.d_model)
        self.patch_norm = (
            nn.LayerNorm(self.patch_len) if self.use_layernorm_patch else nn.Identity()
        )

        self.pos = PositionalEncoding(
            self.d_model, max_len=self.n_patches, dropout=self.dropout
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff,
            dropout=self.dropout, batch_first=True, norm_first=False,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=self.n_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(self.d_model, self.pred_len)
        self.scaler = IqrScaler(quantile_mode)

    def _patchify(self, x):  # x: [B, L, C] -> [B, C, P, patch_len]
        x = x.transpose(1, 2)  # [B, C, L]
        return x.unfold(dimension=2, size=self.patch_len, step=self.stride)

    def forward(self, x):
        B, L, C = x.shape
        patches = self._patchify(x)  # [B, C, P, pl]
        P = patches.size(2)
        patches = self.patch_norm(patches)  # [B, C, P, pl]

        z = patches.reshape(B * C, P, self.patch_len)  # [B*C, P, pl]
        z = self.patch_proj(z)  # [B*C, P, D]
        z = self.pos(z)  # [B*C, P, D]
        z = self.encoder(z)  # [B*C, P, D]

        if self.pool == 'last':
            token = z[:, -1, :]  # [B*C, D]
        else:
            token = z.mean(dim=1)  # [B*C, D]

        yhat = self.head(token)  # [B*C, pred_len]
        yhat = yhat.view(B, C, self.pred_len)  # [B, C, H]
        yhat = yhat.transpose(1, 2)  # [B, H, C]
        return yhat, {}
