from abc import abstractmethod
from nazuna.models.base import BaseModel
from nazuna.scaler import IqrScaler


class BaseScaledModel(BaseModel):
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
