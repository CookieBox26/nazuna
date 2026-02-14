from nazuna.models.base import BasicBaseModel
from nazuna.scaler import IqrScaler
import torch
import torch.nn as nn


class series_decomp(nn.Module):
    def __init__(self, kernel_size, n_moving_avg=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_moving_avg = n_moving_avg
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def moving_avg(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
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


class DLinear(BasicBaseModel):
    """
    !!! note "Original Research"
        This model is based on the following research:
        > Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu.
          "Are Transformers Effective for Time Series Forecasting?"
          In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2023.
          [Paper](https://arxiv.org/abs/2205.13504) |
          [GitHub](https://github.com/cure-lab/LTSF-Linear)
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        kernel_size: int,
        bias: bool,
        quantile_mode_train: str,
        quantile_mode_eval: str,
        n_moving_avg: int = 1,
    ) -> None:
        """
        Args:
            seq_len: Input sequence length
            pred_len: Prediction length
            kernel_size: Kernel size for the moving average decomposition
            bias: Whether to use bias in linear layers
            quantile_mode: Source of quantiles for scaling ('full', 'cum', or 'rolling')
            n_moving_avg: Number of times to apply moving average
        """
        super()._setup(seq_len, pred_len)
        self.decompsition = series_decomp(kernel_size, n_moving_avg)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        self.scaler = IqrScaler(quantile_mode_train, quantile_mode_eval)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x, {'seasonal': seasonal_output, 'trend': trend_output}
