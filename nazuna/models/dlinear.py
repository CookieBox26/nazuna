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
    """
    def _setup(
        self, seq_len: int, pred_len: int, kernel_size: int, bias: bool = True,
        n_moving_avg: int = 1,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_eps: float = 1e-5,
        revin_affine: bool = False, c_in: int | None = None,
        use_lc: bool = False, lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps, revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
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
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int, kernel_size: int, bias: bool,
        n_moving_avg: int = 1,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_eps: float = 1e-5, revin_affine: bool = False,
        use_lc: bool = False, lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps, revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )
        self.decomp = MovingAverageDecomp(kernel_size, n_moving_avg)
        self.seasonal_weight = torch.nn.Parameter(torch.empty(c_in, seq_len, pred_len))
        self.trend_weight = torch.nn.Parameter(torch.empty(c_in, seq_len, pred_len))
        if bias:
            self.seasonal_bias = torch.nn.Parameter(torch.zeros(pred_len, c_in))
            self.trend_bias = torch.nn.Parameter(torch.zeros(pred_len, c_in))
        else:
            self.seasonal_bias = None
            self.trend_bias = None
        w = 1.0 / self.seq_len
        self.seasonal_weight.data.fill_(w)
        self.trend_weight.data.fill_(w)

    def forward(self, x):  # x: [Batch, seq_len, n_channel]
        seasonal_init, trend_init = self.decomp(x)
        seasonal_output = torch.einsum('bsc,csp->bpc', seasonal_init, self.seasonal_weight)
        trend_output = torch.einsum('bsc,csp->bpc', trend_init, self.trend_weight)
        if self.seasonal_bias is not None:
            seasonal_output = seasonal_output + self.seasonal_bias
            trend_output = trend_output + self.trend_bias

        x = seasonal_output + trend_output
        return x, {'seasonal': seasonal_output, 'trend': trend_output}


class DLinearCrossChannel(BasicBaseModel):
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int, kernel_size: int, bias: bool,
        n_moving_avg: int = 1,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_eps: float = 1e-5, revin_affine: bool = False,
        use_lc: bool = False, lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps, revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )
        self.decomp = MovingAverageDecomp(kernel_size, n_moving_avg)
        self.seasonal_weight = torch.nn.Parameter(torch.empty(c_in, c_in, seq_len, pred_len))
        self.trend_weight = torch.nn.Parameter(torch.empty(c_in, c_in, seq_len, pred_len))
        if bias:
            self.seasonal_bias = torch.nn.Parameter(torch.zeros(pred_len, c_in))
            self.trend_bias = torch.nn.Parameter(torch.zeros(pred_len, c_in))
        else:
            self.seasonal_bias = None
            self.trend_bias = None
        w = 1.0 / (seq_len * c_in)
        self.seasonal_weight.data.fill_(w)
        self.trend_weight.data.fill_(w)

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        seasonal_output = torch.einsum('bsc,ocsp->bpo', seasonal_init, self.seasonal_weight)
        trend_output = torch.einsum('bsc,ocsp->bpo', trend_init, self.trend_weight)
        if self.seasonal_bias is not None:
            seasonal_output = seasonal_output + self.seasonal_bias
            trend_output = trend_output + self.trend_bias
        x = seasonal_output + trend_output
        return x, {'seasonal': seasonal_output, 'trend': trend_output}


class DLinearStacked(BasicBaseModel):
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int, kernel_size: int, bias: bool,
        n_moving_avg: int = 1,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_eps: float = 1e-5, revin_affine: bool = False,
        use_lc: bool = False, lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps, revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )
        self.decomp = MovingAverageDecomp(kernel_size, n_moving_avg)
        self.Linear_Seasonal = torch.nn.Linear(seq_len, pred_len, bias=bias)
        self.Linear_Trend = torch.nn.Linear(seq_len, pred_len, bias=bias)
        self.Linear_Seasonal.weight = \
            torch.nn.Parameter(torch.ones(pred_len, seq_len) * (0.9 / seq_len))
        self.Linear_Trend.weight = \
            torch.nn.Parameter(torch.ones(pred_len, seq_len) * (0.9 / seq_len))
        self.seasonal_weight = torch.nn.Parameter(torch.empty(c_in, c_in, seq_len, pred_len))
        self.trend_weight = torch.nn.Parameter(torch.empty(c_in, c_in, seq_len, pred_len))
        if bias:
            self.seasonal_bias = torch.nn.Parameter(torch.zeros(pred_len, c_in))
            self.trend_bias = torch.nn.Parameter(torch.zeros(pred_len, c_in))
        else:
            self.seasonal_bias = None
            self.trend_bias = None
        self.seasonal_weight.data.fill_(0.1 / (seq_len * c_in))
        self.trend_weight.data.fill_(0.1 / (seq_len * c_in))

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
        trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_output_c = torch.einsum('bsc,ocsp->bpo', seasonal_init, self.seasonal_weight)
        trend_output_c = torch.einsum('bsc,ocsp->bpo', trend_init, self.trend_weight)
        if self.seasonal_bias is not None:
            seasonal_output_c = seasonal_output_c + self.seasonal_bias
            trend_output_c = trend_output_c + self.trend_bias

        x = seasonal_output + trend_output + seasonal_output_c + trend_output_c
        return x, {
            'seasonal': seasonal_output + seasonal_output_c,
            'trend': trend_output + trend_output_c,
        }


class NLinear(BasicBaseModel):
    def _setup(
        self, seq_len: int, pred_len: int, bias: bool,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_eps: float = 1e-5,
        revin_affine: bool = False, c_in: int | None = None,
        use_lc: bool = False, lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps, revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )
        self.Linear = torch.nn.Linear(seq_len, pred_len, bias=bias)
        w = 1.0 / seq_len
        self.Linear.weight = torch.nn.Parameter(torch.ones(pred_len, seq_len) * w)

    def forward(self, x):  # x: [B, L, C]
        x_last = x[:, -1:, :]  # [B, 1, C]
        x = x - x_last
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.Linear(x)  # [B, C, T]
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = x + x_last
        return x, {}


class NLinearChannelwise(BasicBaseModel):
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int, bias: bool,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_eps: float = 1e-5, revin_affine: bool = False,
        use_lc: bool = False, lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps, revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )
        self.weight = torch.nn.Parameter(torch.empty(c_in, seq_len, pred_len))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(pred_len, c_in))
        else:
            self.bias = None
        w = 1.0 / seq_len
        self.weight.data.fill_(w)

    def forward(self, x):  # x: [B, L, C]
        x_last = x[:, -1:, :]  # [B, 1, C]
        x = x - x_last
        y = torch.einsum('bsc,csp->bpc', x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y + x_last, {}


class NLinearCrossChannel(BasicBaseModel):
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int, bias: bool,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_eps: float = 1e-5, revin_affine: bool = False,
        use_lc: bool = False, lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps, revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )
        self.weight = torch.nn.Parameter(torch.empty(c_in, c_in, seq_len, pred_len))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(pred_len, c_in))
        else:
            self.bias = None
        w = 1.0 / (seq_len * c_in)
        self.weight.data.fill_(w)

    def forward(self, x):  # x: [B, L, C]
        x_last = x[:, -1:, :]  # [B, 1, C]
        x = x - x_last
        y = torch.einsum('bsc,ocsp->bpo', x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y + x_last, {}


class NLinearPrunedCrossChannel(NLinearCrossChannel):
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int, bias: bool,
        n_ref_channels_target: int, n_epoch_full_ref: int, prune_channels_per_step: int,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_eps: float = 1e-5, revin_affine: bool = False,
        use_lc: bool = False, lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, c_in, bias, scaler_cls, scaler_params, prep_type,
            use_revin=use_revin, revin_eps=revin_eps, revin_affine=revin_affine,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )
        self.c_in = c_in
        self.n_ref_channels_target = n_ref_channels_target
        self.n_epoch_full_ref = n_epoch_full_ref
        self.prune_channels_per_step = prune_channels_per_step
        ref_mask = torch.ones(c_in, c_in, dtype=torch.bool)
        ref_mask.fill_diagonal_(False)
        self.register_buffer('ref_mask', ref_mask)

    def on_epoch_start(self, i_epoch):
        if i_epoch < self.n_epoch_full_ref:
            return
        n_steps = i_epoch - self.n_epoch_full_ref + 1
        n_refs_should_be = self.n_ref_channels_target
        if self.prune_channels_per_step > 0:
            n_refs_should_be = max(
                self.n_ref_channels_target,
                self.c_in - 1 - n_steps * self.prune_channels_per_step,
            )
        with torch.no_grad():
            scores = self.weight.abs().sum(dim=(2, 3))  # [c_out, c_in]
            for o in range(self.c_in):
                active = self.ref_mask[o].clone()
                n_active = active.sum().item()
                if n_active <= n_refs_should_be:
                    continue
                n_prune = n_active - n_refs_should_be
                candidates = torch.where(active)[0]
                cand_scores = scores[o, candidates]
                prune_local = torch.topk(cand_scores, k=n_prune, largest=False).indices
                self.ref_mask[o, candidates[prune_local]] = False

    def forward(self, x):  # x: [B, L, C]
        x_last = x[:, -1:, :]  # [B, 1, C]
        x = x - x_last
        y = torch.einsum('bsc,ocsp->bpo', x, self.weight * self.ref_mask[:, :, None, None])
        if self.bias is not None:
            y = y + self.bias
        return y + x_last, {}


class NLinearStacked(BasicBaseModel):
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int, bias: bool,
        freeze_nlinear: bool = False, nlinear_lr_scale: float = 1.0,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
        prep_type: str = 'none',
        use_revin: bool = False, revin_eps: float = 1e-5, revin_affine: bool = False,
        use_lc: bool = False, lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps, revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )
        self.Linear = torch.nn.Linear(seq_len, pred_len, bias=bias)
        self.Linear.weight = torch.nn.Parameter(torch.ones(pred_len, seq_len) * (0.99 / seq_len))
        self.weight = torch.nn.Parameter(torch.empty(c_in, c_in, seq_len, pred_len))
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(pred_len, c_in))
        self.weight.data.fill_(0.01 / (seq_len * c_in))
        self.freeze_nlinear = freeze_nlinear
        if self.freeze_nlinear:
            for p in self.Linear.parameters():
                p.requires_grad = False
        self.nlinear_lr_scale = nlinear_lr_scale

    def get_args_for_optimizer(self, optimizer_params):
        li_args = []
        if not self.freeze_nlinear:
            args = {'params': (p for p in self.Linear.parameters() if p.requires_grad)}
            args |= optimizer_params
            args['lr'] *= self.nlinear_lr_scale
            li_args.append(args)
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        args = {'params': (p for p in params if p.requires_grad)}
        args |= optimizer_params
        li_args.append(args)
        return li_args

    def forward(self, x):
        x_last = x[:, -1:, :]  # [B, 1, C]
        x = x - x_last
        y = x.permute(0, 2, 1)  # [B, C, L]
        y = self.Linear(y)  # [B, C, T]
        y = y.permute(0, 2, 1)  # [B, T, C]
        z = torch.einsum('bsc,ocsp->bpo', x, self.weight)
        if self.bias is not None:
            z = z + self.bias
        return x_last + y + z, {}
