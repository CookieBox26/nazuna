import torch
from abc import ABC, abstractmethod
from typing import Self, IO, Any
import os
from nazuna.criteria import TimeSeriesError


class BaseModel(torch.nn.Module, ABC):
    """
    Base class for time-series forecasting models.
    """
    def __init__(self, device, **kwargs) -> None:
        super().__init__()
        self.device = device
        self._setup(**kwargs)
        self.to(device)

    @abstractmethod
    def _setup(self, **kwargs) -> None:
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
        Compute loss, set gradients based on batch mean, and return the loss.
        If you want to use custom gradients, override this method.
        """
        loss = self.get_loss(batch, criterion)
        loss.batch_mean.backward()
        return loss

    @classmethod
    def create(
        cls,
        device: str,
        state_path: str | os.PathLike[str] | IO[bytes] = None,
        **kwargs,
    ) -> Self:
        model = cls(device=device, **kwargs)
        if state_path:
            model.load_state_dict(torch.load(state_path, map_location=device))
            model.eval()
        return model

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BasicBaseModel(BaseModel):
    """
    Base class for models that have seq_len and pred_len attributes
    and predict an output sequence from an input sequence.
    """
    def _setup(self, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = None
        self.rescale_loss = False

    def extract_true(self, batch):
        return batch.data_future[:, :self.pred_len]

    def _extract_input(self, batch):
        input_ = batch.data[:, -self.seq_len:, :]
        if self.scaler:
            input_ = self.scaler.scale(input_, batch)
        return input_

    def predict(self, batch):
        input_ = self._extract_input(batch)
        output, info = self.forward(input_)
        if self.scaler:
            output = self.scaler.rescale(output, batch)
        return output, info

    def _get_loss_impl(self, batch, criterion, rescale_loss) -> TimeSeriesError:
        input_ = self._extract_input(batch)
        output, info = self.forward(input_)

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
