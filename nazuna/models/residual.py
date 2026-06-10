from nazuna.models._base import BasicBaseModel
from nazuna.models.common import IqrScaler
from nazuna.criteria import TimeSeriesError
from nazuna.utils import load_class
import torch


def _make_concrete(cls):
    """Create a concrete class from an abstract base class by providing dummy implementations."""
    class ConcreteModel(cls):
        def predict(self, batch):
            raise NotImplementedError('Not used in ResidualModel')
    return ConcreteModel


class ResidualModel(BasicBaseModel):
    """
    Residual learning framework that combines a naive model and a neural model.
    The final prediction is: naive_output + neural_output
    Both sub-models receive the same scaled input and their outputs are summed.
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        naive_model_cls_path: str,
        naive_model_params: dict,
        neural_model_cls_path: str,
        neural_model_params: dict,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
    ) -> None:
        super()._setup(seq_len, pred_len, scaler_cls, scaler_params)
        naive_model_cls = _make_concrete(load_class(naive_model_cls_path))
        self.naive_model = naive_model_cls(device=self.device, **naive_model_params)
        neural_model_cls = _make_concrete(load_class(neural_model_cls_path))
        self.neural_model = neural_model_cls(device=self.device, **neural_model_params)

    def _forward_submodel(self, model, x):
        # Sub-models are called via forward directly, which bypasses their
        # _extract_input/_get_output prep. Apply their diff and RevIN here in
        # the same order (diff outside, RevIN inside) so that e.g. Last4CyclesD
        # averages the differenced series.
        current_value = None
        if model.prep_type == type(model).PrepType.diff:
            current_value = x[:, -1:, :]
            x = x[:, 1:, :] - x[:, :-1, :]
        if model.use_revin:
            x, revin_mean, revin_std = model.revin.normalize(x)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        if model.use_revin:
            out = model.revin.denormalize(out, revin_mean, revin_std)
        if current_value is not None:
            out = current_value + torch.cumsum(out, dim=1)
        return out

    def forward(self, x):
        naive_out = self._forward_submodel(self.naive_model, x)
        neural_out = self._forward_submodel(self.neural_model, x)
        return naive_out + neural_out, {'naive': naive_out}


class ResidualDeseasonModel(ResidualModel):
    """
    Residual framework that feeds the neural model the deseasonalized input.
    The naive model still forecasts on the raw (scaled) input, while the neural
    model receives `x - seasonal(x)` so it focuses on what the seasonal mean
    cannot explain. The final prediction is: naive_output + neural_output.
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        naive_model_cls_path: str,
        naive_model_params: dict,
        neural_model_cls_path: str,
        neural_model_params: dict,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
    ) -> None:
        super()._setup(
            seq_len, pred_len,
            naive_model_cls_path, naive_model_params,
            neural_model_cls_path, neural_model_params,
            scaler_cls, scaler_params,
        )
        self.period_len = period_len

    def _seasonal_reconstruction(self, x):
        # Per-phase mean over the window, tiled back to the input length.
        # Phases are aligned to the end so the last step has phase 0.
        batch_size, seq_len, n_channel = x.shape
        pos = torch.arange(seq_len, device=x.device)
        phase = (seq_len - 1 - pos) % self.period_len
        profile = x.new_zeros(batch_size, self.period_len, n_channel)
        profile.index_add_(1, phase, x)
        counts = x.new_zeros(self.period_len)
        counts.index_add_(0, phase, x.new_ones(seq_len))
        profile = profile / counts.view(1, self.period_len, 1)
        return profile.index_select(1, phase)

    def forward(self, x):
        naive_out = self._forward_submodel(self.naive_model, x)
        residual_in = x - self._seasonal_reconstruction(x)
        neural_out = self._forward_submodel(self.neural_model, residual_in)
        return naive_out + neural_out, {'naive': naive_out}


class ResidualModel1(ResidualModel):
    def get_loss(self, batch, criterion, i_epoch=None) -> TimeSeriesError:
        output, info = self._get_output(batch, False)
        target = self.extract_true(batch)
        output = self.scaler.rescale(output, batch)
        naive = self.scaler.rescale(info['naive'], batch)

        loss_model = criterion(output, target)
        loss_naive = criterion(naive, target)
        loss_model_c = loss_model.each_channel  # batch_size, n_channel
        loss_naive_c = loss_naive.each_channel  # batch_size, n_channel
        penalty_c = torch.clamp(loss_model_c - loss_naive_c, min=0.0)

        alpha = 1.0
        loss_c = loss_model_c + alpha * penalty_c
        loss_model.grad_target = loss_c.mean()
        return loss_model


class ResidualModel2(ResidualModel):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        naive_model_cls_path: str,
        naive_model_params: dict,
        neural_model_cls_path: str,
        neural_model_params: dict,
        n_channel: int,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
    ) -> None:
        super()._setup(
            seq_len, pred_len,
            naive_model_cls_path, naive_model_params,
            neural_model_cls_path, neural_model_params,
            scaler_cls, scaler_params,
        )
        self.w_naive = torch.nn.Parameter(torch.full((n_channel,), 0.5))

    def forward(self, x):
        naive_out = self._forward_submodel(self.naive_model, x)
        neural_out = self._forward_submodel(self.neural_model, x)
        # w_naive: (n_channel,) -> (1, 1, n_channel)
        w = self.w_naive.unsqueeze(0).unsqueeze(0)
        output = w * naive_out + (1 - w) * neural_out
        return output, {'naive': naive_out}


class ResidualModel3(ResidualModel2):
    def get_loss(self, batch, criterion, i_epoch=None) -> TimeSeriesError:
        output, info = self._get_output(batch, False)
        target = self.extract_true(batch)
        output = self.scaler.rescale(output, batch)
        naive = self.scaler.rescale(info['naive'], batch)

        loss_model = criterion(output, target)
        loss_naive = criterion(naive, target)
        loss_model_c = loss_model.each_channel  # batch_size, n_channel
        loss_naive_c = loss_naive.each_channel  # batch_size, n_channel
        penalty_c = torch.clamp(loss_model_c - loss_naive_c, min=0.0)

        alpha = 1.0
        loss_c = loss_model_c + alpha * penalty_c
        loss_model.grad_target = loss_c.mean()
        return loss_model


class ResidualRegularizedModel(ResidualModel):
    """
    Residual framework that keeps the neural part a small correction.
    The prediction is: naive_output + a * neural_output, with a learnable
    scalar weight `a`. The training loss is penalized by both the magnitude of
    the neural correction and the weight `a`, so the neural model is encouraged
    to correct the naive forecast only where it helps.
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        naive_model_cls_path: str,
        naive_model_params: dict,
        neural_model_cls_path: str,
        neural_model_params: dict,
        reg_pred: float = 1e-3,
        reg_coef: float = 1e-3,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
    ) -> None:
        super()._setup(
            seq_len, pred_len,
            naive_model_cls_path, naive_model_params,
            neural_model_cls_path, neural_model_params,
            scaler_cls, scaler_params,
        )
        self.reg_pred = reg_pred
        self.reg_coef = reg_coef
        self.a = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        naive_out = self._forward_submodel(self.naive_model, x)
        neural_out = self._forward_submodel(self.neural_model, x)
        corr = self.a * neural_out
        return naive_out + corr, {'naive': naive_out, 'corr': corr}

    def get_loss(self, batch, criterion, i_epoch=None) -> TimeSeriesError:
        loss = super().get_loss(batch, criterion, i_epoch)
        corr = loss.info['corr']
        penalty = (
            self.reg_pred * corr.pow(2).mean()
            + self.reg_coef * self.a.pow(2)
        )
        loss.grad_target = loss.grad_target + penalty
        return loss


class ResidualRegularizedModel2(ResidualModel2):
    """
    Like ResidualModel2, the naive and neural outputs are mixed with a learnable
    per-channel weight (`w_naive`): w * naive + (1 - w) * neural. The training
    loss combines two penalties: the per-channel "no worse than naive" penalty of
    ResidualModel1/3 (weighted by `reg_worse`), and a small-correction penalty on
    the magnitude of the neural contribution and on the neural weight (1 - w).
    The mixing is thus learned while the neural part stays a small correction over
    the naive forecast.
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        naive_model_cls_path: str,
        naive_model_params: dict,
        neural_model_cls_path: str,
        neural_model_params: dict,
        n_channel: int,
        reg_pred: float = 1e-3,
        reg_coef: float = 1e-3,
        reg_worse: float = 1.0,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
    ) -> None:
        super()._setup(
            seq_len, pred_len,
            naive_model_cls_path, naive_model_params,
            neural_model_cls_path, neural_model_params,
            n_channel, scaler_cls, scaler_params,
        )
        self.reg_pred = reg_pred
        self.reg_coef = reg_coef
        self.reg_worse = reg_worse

    def forward(self, x):
        naive_out = self._forward_submodel(self.naive_model, x)
        neural_out = self._forward_submodel(self.neural_model, x)
        # w_naive: (n_channel,) -> (1, 1, n_channel)
        w = self.w_naive.unsqueeze(0).unsqueeze(0)
        corr = (1 - w) * neural_out
        output = w * naive_out + corr
        return output, {'naive': naive_out, 'corr': corr}

    def get_loss(self, batch, criterion, i_epoch=None) -> TimeSeriesError:
        output, info = self._get_output(batch, False)
        target = self.extract_true(batch)
        output = self.scaler.rescale(output, batch)
        naive = self.scaler.rescale(info['naive'], batch)

        loss_model = criterion(output, target)
        loss_naive = criterion(naive, target)
        loss_model_c = loss_model.each_channel  # batch_size, n_channel
        loss_naive_c = loss_naive.each_channel  # batch_size, n_channel
        penalty_c = torch.clamp(loss_model_c - loss_naive_c, min=0.0)

        loss_c = loss_model_c + self.reg_worse * penalty_c
        reg = (
            self.reg_pred * info['corr'].pow(2).mean()
            + self.reg_coef * (1 - self.w_naive).pow(2).mean()
        )
        loss_model.grad_target = loss_c.mean() + reg
        return loss_model


class ResidualGatedModel(ResidualModel):
    """
    Like ResidualRegularizedModel2 but the naive/neural mixing weight is an
    input-dependent gate g(x) in (0, 1) instead of a static per-channel weight:
    output = g * naive + (1 - g) * neural. The gate is per-sample and per-channel
    (weights shared across channels). The loss uses the same three penalties:
    the per-channel "no worse than naive" penalty (`reg_worse`), the neural
    correction magnitude (`reg_pred`), and the neural gate (1 - g) (`reg_coef`).
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        naive_model_cls_path: str,
        naive_model_params: dict,
        neural_model_cls_path: str,
        neural_model_params: dict,
        reg_pred: float = 1e-3,
        reg_coef: float = 1e-3,
        reg_worse: float = 1.0,
        scaler_cls: type | None = IqrScaler,
        scaler_params: dict | None = {'stat_types': ('qtile_full', 'saved')},
    ) -> None:
        super()._setup(
            seq_len, pred_len,
            naive_model_cls_path, naive_model_params,
            neural_model_cls_path, neural_model_params,
            scaler_cls, scaler_params,
        )
        self.reg_pred = reg_pred
        self.reg_coef = reg_coef
        self.reg_worse = reg_worse
        self.gate = torch.nn.Linear(seq_len, 1)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)

    def _gate(self, x):
        # x: (B, L, C) -> per-sample per-channel gate (B, 1, C) in (0, 1)
        logit = self.gate(x.transpose(1, 2))  # (B, C, 1)
        return torch.sigmoid(logit).transpose(1, 2)

    def forward(self, x):
        naive_out = self._forward_submodel(self.naive_model, x)
        neural_out = self._forward_submodel(self.neural_model, x)
        g = self._gate(x)  # (B, 1, C)
        corr = (1 - g) * neural_out
        output = g * naive_out + corr
        return output, {'naive': naive_out, 'corr': corr, 'gate': g}

    def get_loss(self, batch, criterion, i_epoch=None) -> TimeSeriesError:
        output, info = self._get_output(batch, False)
        target = self.extract_true(batch)
        output = self.scaler.rescale(output, batch)
        naive = self.scaler.rescale(info['naive'], batch)

        loss_model = criterion(output, target)
        loss_naive = criterion(naive, target)
        loss_model_c = loss_model.each_channel  # batch_size, n_channel
        loss_naive_c = loss_naive.each_channel  # batch_size, n_channel
        penalty_c = torch.clamp(loss_model_c - loss_naive_c, min=0.0)

        loss_c = loss_model_c + self.reg_worse * penalty_c
        reg = (
            self.reg_pred * info['corr'].pow(2).mean()
            + self.reg_coef * (1 - info['gate']).pow(2).mean()
        )
        loss_model.grad_target = loss_c.mean() + reg
        return loss_model
