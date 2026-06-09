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
