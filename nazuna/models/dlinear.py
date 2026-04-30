from nazuna.models._base import BasicBaseModel
from nazuna.models.common import IqrScaler, MovingAverageDecomp
import torch


class DLinear(BasicBaseModel):
    """
    !!! info "Original Research"
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
        prep_type = "none"
        use_revin = false
        revin_affine = false
        revin_eps = 1e-5
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
        prep_type: str = 'none',
        use_revin: bool = False, revin_affine: bool = False, revin_eps: float = 1e-5,
        c_in: int | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
        )
        self.decomp = MovingAverageDecomp(kernel_size, n_moving_avg)
        self.Linear_Seasonal = torch.nn.Linear(seq_len, pred_len, bias=bias)
        self.Linear_Trend = torch.nn.Linear(seq_len, pred_len, bias=bias)
        w = 1.0 / seq_len
        self.Linear_Seasonal.weight = torch.nn.Parameter(torch.ones(pred_len, seq_len) * w)
        self.Linear_Trend.weight = torch.nn.Parameter(torch.ones(pred_len, seq_len) * w)

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x, {'seasonal': seasonal_output, 'trend': trend_output}


class DLinearChannelwise(BasicBaseModel):
    """
    !!! tip "Standard parameter settings"
        ```toml
        [definitions.DLinearChannelwise]
        cls_path = "nazuna.models.dlinear.DLinearChannelwise"
        [definitions.DLinearChannelwise.params]
        seq_len = 96  # task-dependent
        pred_len = 24  # task-dependent
        c_in = 7  # task-dependent
        kernel_size = 25
        bias = true
        n_moving_avg = 1
        scaler_cls_path = "nazuna.models.common.IqrScaler"
        scaler_params = { "stat_types" = [ "qtile_full", "saved",] }
        prep_type = "none"
        use_revin = false
        revin_affine = false
        revin_eps = 1e-5
        ```
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        kernel_size: int,
        bias: bool,
        n_moving_avg: int = 1,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_affine: bool = False, revin_eps: float = 1e-5,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
        )
        self.decomp = MovingAverageDecomp(kernel_size, n_moving_avg)
        self.seasonal_weight = torch.nn.Parameter(torch.empty(c_in, seq_len, pred_len))
        self.trend_weight = torch.nn.Parameter(torch.empty(c_in, seq_len, pred_len))
        if bias:
            self.seasonal_bias = torch.nn.Parameter(torch.zeros(c_in, pred_len))
            self.trend_bias = torch.nn.Parameter(torch.zeros(c_in, pred_len))
        else:
            self.seasonal_bias = None
            self.trend_bias = None
        w = 1.0 / self.seq_len
        self.seasonal_weight.data.fill_(w)
        self.trend_weight.data.fill_(w)

    def forward(self, x):
        # x: [Batch, seq_len, n_channel]
        seasonal_init, trend_init = self.decomp(x)

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


class NLinear(BasicBaseModel):
    """
    !!! tip "Standard parameter settings"
        ```toml
        [definitions.NLinear]
        cls_path = "nazuna.models.dlinear.NLinear"
        [definitions.NLinear.params]
        seq_len = 96  # task-dependent
        pred_len = 24  # task-dependent
        bias = true
        scaler_cls_path = "nazuna.models.common.IqrScaler"
        scaler_params = { "stat_types" = [ "qtile_full", "saved",] }
        prep_type = "none"
        use_revin = false
        revin_affine = false
        revin_eps = 1e-5
        ```
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        bias: bool,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_affine: bool = False, revin_eps: float = 1e-5,
        c_in: int | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
        )
        self.Linear = torch.nn.Linear(seq_len, pred_len, bias=bias)
        w = 1.0 / seq_len
        self.Linear.weight = torch.nn.Parameter(torch.ones(pred_len, seq_len) * w)

    def forward(self, x):
        # x: [B, L, C]
        x_last = x[:, -1:, :]  # [B, 1, C]
        x = x - x_last
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.Linear(x)  # [B, C, T]
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = x + x_last
        return x, {}


class NLinearChannelwise(BasicBaseModel):
    """
    !!! tip "Standard parameter settings"
        ```toml
        [definitions.NLinearChannelwise]
        cls_path = "nazuna.models.dlinear.NLinearChannelwise"
        [definitions.NLinearChannelwise.params]
        seq_len = 96  # task-dependent
        pred_len = 24  # task-dependent
        c_in = 7  # task-dependent
        bias = true
        scaler_cls_path = "nazuna.models.common.IqrScaler"
        scaler_params = { "stat_types" = [ "qtile_full", "saved",] }
        prep_type = "none"
        use_revin = false
        revin_affine = false
        revin_eps = 1e-5
        ```
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        bias: bool,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_affine: bool = False, revin_eps: float = 1e-5,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
        )
        self.weight = torch.nn.Parameter(torch.empty(c_in, seq_len, pred_len))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(c_in, pred_len))
        else:
            self.bias = None
        w = 1.0 / seq_len
        self.weight.data.fill_(w)

    def forward(self, x):
        # x: [B, L, C]
        x_last = x[:, -1:, :]  # [B, 1, C]
        x = x - x_last
        x = x.permute(0, 2, 1)  # [B, C, L]
        # einsum: (B, C, S) x (C, S, P) -> (B, C, P)
        x = torch.einsum('bcs,csp->bcp', x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = x + x_last
        return x, {}
