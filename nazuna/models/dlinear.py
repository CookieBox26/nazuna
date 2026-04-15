from nazuna.models._base import BasicBaseModel
from nazuna.models.common import SeriesDecomp
from nazuna.scaler import IqrScaler
import torch
import math


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
        self.decompsition = SeriesDecomp(kernel_size, n_moving_avg)
        self.Linear_Seasonal = torch.nn.Linear(self.seq_len, self.pred_len, bias=bias)
        self.Linear_Trend = torch.nn.Linear(self.seq_len, self.pred_len, bias=bias)
        self.scaler = IqrScaler(quantile_mode_train, quantile_mode_eval)
        self._init_weights()

    def _init_weights(self):
        val = 1.0 / self.seq_len
        self.Linear_Seasonal.weight = torch.nn.Parameter(
            torch.ones(self.pred_len, self.seq_len) * val
        )
        self.Linear_Trend.weight = torch.nn.Parameter(
            torch.ones(self.pred_len, self.seq_len) * val
        )

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x, {'seasonal': seasonal_output, 'trend': trend_output}


class NLinear(BasicBaseModel):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        bias: bool,
        quantile_mode_train: str,
        quantile_mode_eval: str,
    ) -> None:
        super()._setup(seq_len, pred_len)
        self.Linear = torch.nn.Linear(self.seq_len, self.pred_len, bias=bias)
        self.scaler = IqrScaler(quantile_mode_train, quantile_mode_eval)
        self._init_weights()

    def _init_weights(self):
        val = 1.0 / self.seq_len
        self.Linear.weight = torch.nn.Parameter(
            torch.ones(self.pred_len, self.seq_len) * val
        )

    def forward(self, x):
        # x: [B, L, C]
        x_last = x[:, -1:, :]  # [B, 1, C]
        x = x - x_last
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.Linear(x)  # [B, C, T]
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = x + x_last
        return x, {}


class DLinearChannelwise(BasicBaseModel):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        n_channel: int,
        kernel_size: int,
        bias: bool,
        quantile_mode_train: str,
        quantile_mode_eval: str,
        n_moving_avg: int = 1,
    ) -> None:
        super()._setup(seq_len, pred_len)
        self.n_channel = n_channel
        self.decompsition = SeriesDecomp(kernel_size, n_moving_avg)

        # [n_channel, seq_len, pred_len]
        self.seasonal_weight = torch.nn.Parameter(
            torch.empty(n_channel, seq_len, pred_len)
        )
        self.trend_weight = torch.nn.Parameter(
            torch.empty(n_channel, seq_len, pred_len)
        )
        torch.nn.init.kaiming_uniform_(self.seasonal_weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.trend_weight, a=math.sqrt(5))
        if bias:
            self.seasonal_bias = torch.nn.Parameter(torch.zeros(n_channel, pred_len))
            self.trend_bias = torch.nn.Parameter(torch.zeros(n_channel, pred_len))
        else:
            self.seasonal_bias = None
            self.trend_bias = None

        self.scaler = IqrScaler(quantile_mode_train, quantile_mode_eval)
        self._init_weights()

    def _init_weights(self):
        val = 1.0 / self.seq_len
        self.seasonal_weight.data.fill_(val)
        self.trend_weight.data.fill_(val)

    def forward(self, x):
        # x: [Batch, seq_len, n_channel]
        seasonal_init, trend_init = self.decompsition(x)

        # [Batch, n_channel, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # einsum: (B, C, S) x (C, S, P) -> (B, C, P)
        seasonal_output = torch.einsum('bcs,csp->bcp', seasonal_init, self.seasonal_weight)
        trend_output = torch.einsum('bcs,csp->bcp', trend_init, self.trend_weight)
        if self.seasonal_bias is not None:
            seasonal_output = seasonal_output + self.seasonal_bias
            trend_output = trend_output + self.trend_bias

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, pred_len, n_channel]
        return x, {'seasonal': seasonal_output, 'trend': trend_output}
