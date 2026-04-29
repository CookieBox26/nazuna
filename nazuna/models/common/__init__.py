from nazuna.models.common.scalers import IqrScaler as IqrScaler
from nazuna.models.common.time_feature_embedding import (
    TimeFeatureEmbedding as TimeFeatureEmbedding,
)
from nazuna.models.common.transformer import (
    MultiheadAttention as MultiheadAttention,
    TransformerEncoderLayer as TransformerEncoderLayer,
)
import torch


class RevIN(torch.nn.Module):
    def __init__(self, c_in: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.c_in = c_in
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.affine_weight = torch.nn.Parameter(torch.ones(c_in))
            self.affine_bias = torch.nn.Parameter(torch.zeros(c_in))

    def normalize(self, x):
        x_mean = x.mean(dim=1, keepdim=True).detach()  # [B, 1, C]
        x_var = x.var(dim=1, keepdim=True, unbiased=False)  # [B, 1, C]
        x_std = torch.sqrt(x_var + self.eps).detach()  # [B, 1, C]
        x_normalized = (x - x_mean) / x_std
        if self.affine:
            x_normalized = x_normalized * self.affine_weight + self.affine_bias
        return x_normalized, x_mean, x_std

    def denormalize(self, y, x_mean, x_std):
        if self.affine:
            y = (y - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        return y * x_std + x_mean


class SeriesDecomp(torch.nn.Module):
    """Moving average based series decomposition.

    Pads the input by repeating edge values, then applies average pooling.
    For odd kernel_size, pads ``(kernel_size - 1) // 2`` steps on each side.
    For even kernel_size, pads ``kernel_size // 2`` steps on the front and
    ``(kernel_size - 1) // 2`` steps on the end (one extra step at the front).
    """
    def __init__(self, kernel_size, n_moving_avg=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_moving_avg = n_moving_avg
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def moving_avg(self, x):
        pad_len_front = self.kernel_size // 2
        pad_len_end = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len_front, 1)
        end = x[:, -1:, :].repeat(1, pad_len_end, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        moving_mean = x
        for _ in range(self.n_moving_avg):
            moving_mean = self.moving_avg(moving_mean)
        res = x - moving_mean
        return res, moving_mean


class BatchSeriesNorm(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm1d(d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        return x
