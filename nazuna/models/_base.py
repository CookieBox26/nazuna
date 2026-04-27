from abc import ABC, abstractmethod
from nazuna.criteria import TimeSeriesError
from enum import Enum
from typing import Self, IO, Any
import torch
import os


class BaseModel(torch.nn.Module, ABC):
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

    def get_loss_and_backward(self, batch, criterion) -> TimeSeriesError:
        """
        Compute loss, set gradients based on target (default: batch mean),
        and return the loss.
        """
        loss = self.get_loss(batch, criterion)
        loss.get_grad_target().backward()
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

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BasicBaseModel(BaseModel):
    """
    Base class for models that have seq_len and pred_len attributes
    and predict an output sequence from an input sequence.
    """
    # preprocessing type (applied after scaling)
    PrepType = Enum('PrepType', ['none', 'diff'])

    def _setup(
        self, seq_len, pred_len,
        scaler_cls=None, scaler_params=None, rescale_loss=True,
        prep_type: str = 'none',
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

    def extract_true(self, batch):
        return batch.data_future[:, :self.pred_len]

    def _extract_input(self, batch):
        input_ = batch.data[:, -self.seq_len_required:]
        if self.scaler:
            input_ = self.scaler.scale(input_, batch)
        current_value = None  # used only for inverse when differencing
        if self.prep_type == type(self).PrepType.diff:
            current_value = input_[:, -1:, :]
            input_ = input_[:, 1:, :] - input_[:, :-1, :]
        return input_, current_value

    def _get_output(self, batch, rescale):
        input_, current_value = self._extract_input(batch)
        output, info = self.forward(input_)
        if self.prep_type == type(self).PrepType.diff:
            output = current_value + torch.cumsum(output, dim=1)
        if rescale:
            output = self.scaler.rescale(output, batch)
        return output, info

    def predict(self, batch):
        return self._get_output(batch, (self.scaler is not None))

    def _get_loss_impl(self, batch, criterion, rescale_loss) -> TimeSeriesError:
        output, info = self._get_output(batch, False)
        target = self.extract_true(batch)
        if self.scaler:
            if rescale_loss:  # compute loss in original space
                output = self.scaler.rescale(output, batch)
            else:  # compute loss in scaled space
                target = self.scaler.scale(target, batch)

        loss = criterion(output, target)
        loss.info.update(info)
        return loss

    def get_loss(self, batch, criterion) -> TimeSeriesError:
        return self._get_loss_impl(batch, criterion, self.rescale_loss)
