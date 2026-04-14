from nazuna.models.base import BasicBaseModel
from nazuna.scaler import IqrScaler
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(
            torch.zeros(max_len, d_model)
        )  # [max_len, d_model]

    def forward(self, x):  # x: [B, S, D]
        s = x.size(1)
        return self.dropout(x + self.pe[:s].unsqueeze(0))


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout, batch_first=True,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.BatchNorm1d(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.BatchNorm1d(d_model)

    def forward(self, x):  # x: [B, S, D]
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout_attn(attn_out)
        # BatchNorm1d expects [B, D, S]
        x = self.norm_attn(x.transpose(1, 2)).transpose(1, 2)

        x = x + self.ff(x)
        x = self.norm_ff(x.transpose(1, 2)).transpose(1, 2)
        return x


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
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        revin: bool = True,
        revin_eps: float = 1e-5,
        quantile_mode_train: str | None = None,
        quantile_mode_eval: str | None = None,
    ) -> None:
        """
        Args:
            seq_len: Input sequence length (must be >= `patch_len`)
            pred_len: Prediction length
            patch_len: Length of each patch
            stride: Stride for patch extraction
            d_model: Dimension of the model
            n_heads: Number of attention heads
            n_layers: Number of Transformer encoder layers
            d_ff: Dimension of the feedforward network
            dropout: Dropout rate
            revin: Whether to apply RevIN (instance normalization)
            revin_eps: Epsilon for RevIN std computation
            quantile_mode_train: IqrScaler mode for training (None to disable)
            quantile_mode_eval: IqrScaler mode for evaluation (None to disable)
        """
        super()._setup(seq_len, pred_len)

        if quantile_mode_train is not None:
            self.scaler = IqrScaler(
                quantile_mode_train, quantile_mode_eval,
            )

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_revin = revin
        self.revin_eps = revin_eps
        self.pool = 'last'  # 'last' or 'mean'

        assert seq_len >= self.patch_len, 'seq_len >= patch_len'
        self.n_patches = (
            1 + (self.seq_len - self.patch_len) // self.stride
        )

        self.patch_proj = nn.Linear(self.patch_len, self.d_model)

        self.pos = PositionalEncoding(
            self.d_model,
            max_len=self.n_patches,
            dropout=self.dropout,
        )

        self.encoder = nn.Sequential(*[
            TSTEncoderLayer(
                self.d_model, self.n_heads,
                self.d_ff, self.dropout,
            )
            for _ in range(self.n_layers)
        ])
        self.head = nn.Linear(self.d_model, self.pred_len)

    def _patchify(self, x):  # x: [B, L, C] -> [B, C, P, patch_len]
        x = x.transpose(1, 2)  # [B, C, L]
        return x.unfold(dimension=2, size=self.patch_len, step=self.stride)

    def forward(self, x):
        B, L, C = x.shape

        # RevIN: instance normalization (per-sample, per-channel)
        if self.use_revin:
            # x: [B, L, C]
            ri_mean = x.mean(dim=1, keepdim=True)  # [B, 1, C]
            ri_std = (x.std(dim=1, keepdim=True) + self.revin_eps)  # [B, 1, C]
            x = (x - ri_mean) / ri_std

        patches = self._patchify(x)  # [B, C, P, pl]
        P = patches.size(2)
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

        # RevIN: de-normalize
        if self.use_revin:
            yhat = yhat * ri_std + ri_mean

        return yhat, {}


class DiffPatchTST(PatchTST):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        revin: bool = True,
        revin_eps: float = 1e-5,
        quantile_mode_train: str | None = None,
        quantile_mode_eval: str | None = None,
    ) -> None:
        # After first-order differencing, length becomes seq_len - 1.
        diff_seq_len = seq_len - 1
        super()._setup(
            diff_seq_len, pred_len,
            patch_len, stride, d_model, n_heads,
            n_layers, d_ff, dropout,
            revin, revin_eps,
            quantile_mode_train, quantile_mode_eval,
        )
        # Restore original seq_len for _extract_input slicing.
        self.seq_len = seq_len

    def forward(self, x):  # x: [B, seq_len, C] (scaled)
        last_val = x[:, -1:, :]  # [B, 1, C]
        dx = x[:, 1:, :] - x[:, :-1, :]  # [B, seq_len-1, C]
        pred_dx, info = super().forward(dx)  # [B, pred_len, C]
        pred = last_val + torch.cumsum(pred_dx, dim=1)
        return pred, info
