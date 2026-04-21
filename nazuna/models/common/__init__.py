from nazuna.models.common.scalers import IqrScaler as IqrScaler
from nazuna.models.common.time_feature_embedding \
    import TimeFeatureEmbedding as TimeFeatureEmbedding
import torch


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
