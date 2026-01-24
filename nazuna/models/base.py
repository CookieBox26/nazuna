import torch
from abc import ABC, abstractmethod


class BaseModel(torch.nn.Module, ABC):
    """
    Base class for time-series forecasting models.
    Subclasses must implement _setup() to construct their layers.
    The device is handled by this base class, so subclasses don't need to manage it.
    """
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device
        self._setup(**kwargs)
        self.to(device)

    @abstractmethod
    def _setup(self, **kwargs):
        """
        Construct layers.
        """
        pass

    @classmethod
    def create(cls, device, state_path=None, **kwargs):
        model = cls(device=device, **kwargs)
        if state_path:
            model.load_state_dict(torch.load(state_path, map_location=device))
            model.eval()
        return model

    @abstractmethod
    def get_loss(self, batch, criterion):
        """
        Must return a TimeSeriesLoss object.
        """
        pass

    def get_loss_and_backward(self, batch, criterion):
        """
        """
        loss = self.get_loss(batch, criterion)
        loss.batch_mean.backward()
        return loss

    @abstractmethod
    def predict(self, batch):
        pass

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseModelWithScaler(BaseModel):
    """
    Base class for models that use IQR scaling with quantiles.

    Subclasses must:
    - Call `self._setup_scaler(quantile_mode)` in their `_setup()` method
    - Set `self.seq_len` and `self.pred_len` before calling `_setup_scaler()`
    - Implement `forward(x)` which returns either a tensor or a tuple (tensor, info_dict)
    """

    def _setup_scaler(self, quantile_mode: str) -> None:
        """
        Initialize the scaler and quantile mode.

        Args:
            quantile_mode: Source of quantiles for scaling ('full', 'cum', or 'rolling')
        """
        from nazuna.scaler import IqrScaler
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
            raise ValueError(f'Unknown quantile_mode: {self.quantile_mode}')
        q1s_ = quantiles[:, 0, :]
        q2s_ = quantiles[:, 1, :]
        q3s_ = quantiles[:, 2, :]
        return q1s_, q2s_, q3s_

    def _extract_output(self, output):
        """Extract the prediction tensor from forward() output."""
        if isinstance(output, tuple):
            return output[0]
        return output

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
        input_ = self.extract_input(batch)
        output = self._extract_output(self(input_))
        return self.scaler.rescale(output, q1s_=q1s_, q2s_=q2s_, q3s_=q3s_)

    def get_loss(self, batch, criterion):
        input_ = self.extract_input(batch)
        target = self.extract_target(batch)
        output = self._extract_output(self(input_))
        return criterion(output, target)

    @abstractmethod
    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Scaled input tensor [batch_size, seq_len, n_channel]

        Returns:
            Either a tensor [batch_size, pred_len, n_channel] or
            a tuple (tensor, info_dict)
        """
        pass
