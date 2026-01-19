from abc import abstractmethod
from nazuna.models.base import BaseModel
from nazuna.scaler import IqrScaler
import torch
import torch.nn as nn


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def moving_avg(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class BaseDLinear(BaseModel):
    def _init_layers(self, seq_len, pred_len, kernel_size, bias):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len, bias=bias)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x, {'seasonal': seasonal_output, 'trend': trend_output}

    @abstractmethod
    def extract_input(self, batch):
        pass

    @abstractmethod
    def extract_target(self, batch):
        pass

    @abstractmethod
    def predict(self, batch):
        pass

    def get_loss(self, batch, criterion):
        input_ = self.extract_input(batch)
        target = self.extract_target(batch)
        output, _ = self(input_)
        loss = criterion(output, target)
        return loss, {}


class DLinear(BaseDLinear):
    """
    !!! note "Original Research"
        This model is based on the following research:
        > Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu.
          "Are Transformers Effective for Time Series Forecasting?"
          In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2023.
          [Paper](https://arxiv.org/abs/2205.13504) |
          [GitHub](https://github.com/cure-lab/LTSF-Linear)
    """
    def _init_layers(self, seq_len, pred_len, kernel_size, bias, quantile_mode):
        """
        Args:
            seq_len: Input sequence length
            pred_len: Prediction length
            kernel_size: Kernel size for the moving average decomposition
            bias: Whether to use bias in linear layers
            quantile_mode: Source of quantiles for scaling ('full', 'cum', or 'rolling')
        """
        super()._init_layers(seq_len, pred_len, kernel_size, bias)
        self.quantile_mode = quantile_mode
        self.scaler = IqrScaler()

    def _get_quantiles(self, batch):
        if self.quantile_mode == 'full':
            quantiles = batch.quantiles_full
        elif self.quantile_mode == 'cum':
            quantiles = batch.quantiles_cum
        elif self.quantile_mode == 'rolling':
            quantiles = batch.quantiles_rolling
        else:
            raise ValueError(f"Unknown quantile_mode: {self.quantile_mode}")
        q1s_ = quantiles[:, 0, :]
        q2s_ = quantiles[:, 1, :]
        q3s_ = quantiles[:, 2, :]
        return q1s_, q2s_, q3s_

    def extract_input(self, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        return self.scaler.scale(
            batch.data[:, -self.seq_len:, :], q1s_=q1s_, q2s_=q2s_, q3s_=q3s_
        )

    def extract_target(self, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        return self.scaler.scale(
            batch.data_future[:, :self.pred_len], q1s_=q1s_, q2s_=q2s_, q3s_=q3s_
        )

    def predict(self, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        input = self.extract_input(batch)
        output, _ = self(input)
        return self.scaler.rescale(output, q1s_=q1s_, q2s_=q2s_, q3s_=q3s_)
