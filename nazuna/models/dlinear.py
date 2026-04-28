from nazuna.models._base import BasicBaseModel
from nazuna.models.common import IqrScaler, SeriesDecomp
import torch
import math


class DLinear(BasicBaseModel):
    """
    !!! note "Original Research"
        This model is based on the following research:
        > Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu.
          "Are Transformers Effective for Time Series Forecasting?"
          In Proceedings of the AAAI Conference on Artificial Intelligence
          (AAAI 2023), vol. 37, pp. 11121-11128, 2023.
          [arXiv](https://arxiv.org/abs/2205.13504) |
          [GitHub](https://github.com/cure-lab/LTSF-Linear)

    !!! tip "Standard parameter settings"
        ```toml
        [definitions.DLinear]
        cls_path = "nazuna.models.dlinear.DLinear"
        [definitions.DLinear.params]
        seq_len = 96  # task-dependent
        pred_len = 24  # task-dependent
        kernel_size = 25
        bias = true
        n_moving_avg = 1
        scaler_cls_path = "nazuna.models.common.IqrScaler"
        scaler_params = { "stat_types" = [ "qtile_full", "saved",] }
        ```
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        kernel_size: int,
        bias: bool = True,
        n_moving_avg: int = 1,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
    ) -> None:
        super()._setup(seq_len, pred_len, scaler_cls, scaler_params)
        self.decompsition = SeriesDecomp(kernel_size, n_moving_avg)
        self.Linear_Seasonal = torch.nn.Linear(self.seq_len, self.pred_len, bias=bias)
        self.Linear_Trend = torch.nn.Linear(self.seq_len, self.pred_len, bias=bias)
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
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
    ) -> None:
        super()._setup(seq_len, pred_len, scaler_cls, scaler_params)
        self.Linear = torch.nn.Linear(self.seq_len, self.pred_len, bias=bias)
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
        n_moving_avg: int = 1,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
    ) -> None:
        super()._setup(seq_len, pred_len, scaler_cls, scaler_params)
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
