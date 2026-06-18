from abc import ABC, abstractmethod
from nazuna.criteria import TimeSeriesError
from nazuna.models.common import RevIN
from enum import Enum
from typing import Self, IO, Any
import torch
import os


class BaseModel(torch.nn.Module, ABC):
    optimization_part_names = ['model']

    """
    Base class for time-series forecasting models.
    """
    def __init__(self, device, **setup_args) -> None:
        super().__init__()
        self.device = device
        self._setup(**setup_args)
        self.to(device)

    @abstractmethod
    def _setup(self, **setup_args) -> None:
        """
        Define required hyperparameters and construct layers.
        """
        pass

    @abstractmethod
    def extract_true(self, batch) -> Any:
        """
        Extract the ground truth tensor from the batch.
        """
        pass

    @abstractmethod
    def _extract_input(self, batch) -> Any:
        """
        Extract only required inference inputs from the batch, apply scaling if needed.
        """
        pass

    @abstractmethod
    def forward(self, input_) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Take extracted input and return output tensor with debug info dict.
        The output tensor is expected to be before rescaling.
        """
        pass

    @abstractmethod
    def predict(self, batch) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Take a batch and return predicted tensor with debug info dict.
        If the model directly outputs the prediction, simply:
        ```py
        input_ = self._extract_input(batch)
        output, info = self.forward(input_)
        return output, info
        ```
        If forward operates in scaled space, rescaling output is required.
        """
        pass

    @abstractmethod
    def get_loss(self, batch, criterion) -> TimeSeriesError:
        """
        Compute and return the loss given a batch and criterion.
        If the model directly outputs predictions and simply minimizes error:
        ```py
        input_ = self._extract_input(batch)
        true = self._extract_true(batch)
        output, info = self.forward(input_)
        loss = criterion(output, true)
        loss.info.update(info)
        return loss
        ```
        If forward operates in scaled space, either rescale output
        (to compute loss in original space) or scale true
        (to compute loss in scaled space).
        """
        pass

    def get_loss_and_backward(
        self, batch, criterion, i_epoch=None, i_batch=None,
    ) -> TimeSeriesError:
        """
        Compute loss, set gradients based on target (default: batch mean),
        and return the loss.
        """
        loss = self.get_loss(batch, criterion, i_epoch, i_batch)
        loss.grad_target.backward()
        return loss

    @classmethod
    def create(
        cls,
        device: str,
        state_path: str | os.PathLike[str] | IO[bytes] = None,
        **setup_args,
    ) -> Self:
        model = cls(device=device, **setup_args)
        if state_path:
            state_dict = torch.load(state_path, map_location=device)
            if hasattr(model, 'scaler') and model.scaler:
                model.scaler.prepare_load_state_dict(state_dict)
            model.load_state_dict(state_dict, strict=False)
        return model

    @classmethod
    def validate_optimizer_groups(cls, optimizer_groups):
        groups = optimizer_groups.groups
        if set(groups) == {'model'}:  # Always allow the single key 'model'
            return
        assert set(groups) == set(cls.optimization_part_names), \
            'optimizer_groups keys are inconsistent'

    def set_optimizers(self, optimizer_groups):
        params = (p for p in self.parameters() if p.requires_grad)
        optimizer_groups.set_optimizer('model', params)

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BasicBaseModel(BaseModel):
    """
    Base class for models that have seq_len and pred_len attributes
    and predict an output sequence from an input sequence.
    """
    PrepType = Enum('PrepType', ['none', 'diff'])  # preprocessing type (applied after scaling)

    def __init__(self, device, **setup_args) -> None:
        setup_args = type(self)._resolve_seq_len(setup_args)
        super().__init__(device, **setup_args)

    @staticmethod
    def _resolve_seq_len(args):
        seq_len = args.get('seq_len')
        seq_len_raw = args.pop('seq_len_raw', None)
        # Treat negative values as unspecified (so templates can default to e.g. -1).
        if seq_len is not None and seq_len < 0:
            seq_len = None
        if seq_len_raw is not None and seq_len_raw < 0:
            seq_len_raw = None
        prep_type = args.get('prep_type', 'none')
        delta = 1 if prep_type == 'diff' else 0
        if seq_len is None and seq_len_raw is None:
            raise ValueError('Either seq_len or seq_len_raw must be specified.')
        if seq_len is None:
            args['seq_len'] = seq_len_raw - delta
        elif seq_len_raw is not None and seq_len + delta != seq_len_raw:
            raise ValueError(
                f'seq_len_raw ({seq_len_raw}) must equal '
                f'seq_len ({seq_len}) + {delta} for prep_type={prep_type!r}.'
            )
        return args

    def _setup(
        self, seq_len, pred_len,
        scaler_cls=None, scaler_params=None, rescale_loss=True,
        prep_type: str = 'none',
        use_revin=False, revin_eps=1e-5, revin_affine=False, c_in=None,
        use_lc=False, lc_end_epoch=20, lc_rate=0.9,  # loss curriculum
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = None
        if scaler_cls is not None:
            self.scaler = scaler_cls(**scaler_params)
        self.rescale_loss = rescale_loss
        self.prep_type = type(self).PrepType[prep_type]
        self.seq_len_required = self.seq_len
        if self.prep_type == type(self).PrepType.diff:
            self.seq_len_required += 1
        self.use_revin = use_revin
        if self.use_revin:
            self.revin = RevIN(eps=revin_eps, affine=revin_affine, c_in=c_in)
        self.use_lc = use_lc
        self.lc_end_epoch = lc_end_epoch
        self.lc_rate = lc_rate

    def extract_true(self, batch):
        return batch.data_future[:, :self.pred_len]

    def _extract_input(self, batch):
        prep_info = {}
        input_ = batch.data[:, -self.seq_len_required:]
        if self.scaler:
            input_ = self.scaler.scale(input_, batch)
        if self.prep_type == type(self).PrepType.diff:
            prep_info['current_value'] = input_[:, -1:, :]
            input_ = input_[:, 1:, :] - input_[:, :-1, :]
        if self.use_revin:
            input_, revin_mean, revin_std = self.revin.normalize(input_)
            prep_info['revin_mean'] = revin_mean
            prep_info['revin_std'] = revin_std
        return input_, prep_info

    def _get_output(self, batch, rescale):
        input_, prep_info = self._extract_input(batch)
        output, info = self.forward(input_)
        if self.use_revin:
            revin_mean = prep_info['revin_mean']
            revin_std = prep_info['revin_std']
            output = self.revin.denormalize(output, revin_mean, revin_std)
        if self.prep_type == type(self).PrepType.diff:
            output = prep_info['current_value'] + torch.cumsum(output, dim=1)
        if rescale:
            output = self.scaler.rescale(output, batch)
        return output, info

    def predict(self, batch):
        return self._get_output(batch, (self.scaler is not None))

    def get_loss(self, batch, criterion, i_epoch=None, i_batch=None) -> TimeSeriesError:
        output, info = self._get_output(batch, False)
        target = self.extract_true(batch)
        if self.scaler:
            if self.rescale_loss:  # compute loss in original space
                output = self.scaler.rescale(output, batch)
            else:  # compute loss in scaled space
                target = self.scaler.scale(target, batch)

        loss = criterion(output, target)
        if self.use_lc and i_epoch < self.lc_end_epoch:
            d = self.lc_end_epoch - i_epoch
            r = (criterion.decay_rate or 1.0) * (self.lc_rate ** d)
            idx = torch.arange(self.pred_len, dtype=torch.float, device=self.device)
            w = torch.pow(torch.tensor(r, dtype=torch.float, device=self.device), idx)
            w_seq = w / w.sum()
            error_channel = torch.einsum('j,ijk->ik', (w_seq, loss.each_point))
            error_sample = torch.einsum('k,ik->i', (criterion.w_channel, error_channel))
            loss.grad_target = error_sample.mean()
        else:
            loss.grad_target = loss.each_sample.mean()
        loss.info.update(info)
        return loss
