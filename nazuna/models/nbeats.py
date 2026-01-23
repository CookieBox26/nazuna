from abc import abstractmethod
from nazuna.models.base import BaseModel
from nazuna.scaler import IqrScaler
import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    """Basic building block of N-BEATS."""

    def __init__(self, input_size, theta_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.theta_b = nn.Linear(hidden_size, theta_size, bias=False)
        self.theta_f = nn.Linear(hidden_size, theta_size, bias=False)
        self.backcast = nn.Linear(theta_size, input_size, bias=False)
        self.forecast = nn.Linear(theta_size, output_size, bias=False)

    def forward(self, x):
        h = self.fc(x)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        backcast = self.backcast(theta_b)
        forecast = self.forecast(theta_f)
        return backcast, forecast


class BaseNBEATS(BaseModel):
    def _setup(self, seq_len, pred_len, n_stacks, n_blocks, hidden_size, theta_size):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.theta_size = theta_size

        self.blocks = nn.ModuleList()
        for _ in range(n_stacks):
            for _ in range(n_blocks):
                self.blocks.append(
                    NBeatsBlock(
                        input_size=seq_len,
                        theta_size=theta_size,
                        hidden_size=hidden_size,
                        output_size=pred_len,
                    )
                )

    def forward(self, x):
        # x: [B, L, C] -> channel-independent processing
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L)  # [B*C, L]

        residual = x
        forecast = torch.zeros(B * C, self.pred_len, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        forecast = forecast.view(B, C, self.pred_len)  # [B, C, H]
        forecast = forecast.permute(0, 2, 1)  # [B, H, C]
        return forecast

    @abstractmethod
    def extract_input(self, batch):
        pass

    @abstractmethod
    def extract_target(self, batch):
        pass

    @abstractmethod
    def predict(self, batch):
        pass

    def get_loss(self, batch, criterion):
        input_ = self.extract_input(batch)
        target = self.extract_target(batch)
        output = self(input_)
        return criterion(output, target)


class NBEATS(BaseNBEATS):
    """
    !!! note "Original Research"
        This model is based on the following research:
        > Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio.
          "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting."
          In International Conference on Learning Representations (ICLR), 2020.
          [Paper](https://arxiv.org/abs/1905.10437) |
          [GitHub](https://github.com/ServiceNow/N-BEATS)

    !!! note "Reference Implementations"
        - Official: https://github.com/ServiceNow/N-BEATS
        - Community: https://github.com/philipperemy/n-beats

    Note:
        This implementation uses the generic (non-interpretable) architecture
        with channel-independent processing for multivariate time series.
    """

    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        quantile_mode: str,
        n_stacks: int = 2,
        n_blocks: int = 3,
        hidden_size: int = 256,
        theta_size: int = 32,
    ) -> None:
        """
        Args:
            seq_len: Input sequence length
            pred_len: Prediction length
            quantile_mode: Source of quantiles for scaling ('full', 'cum', or 'rolling')
            n_stacks: Number of stacks (default: 2)
            n_blocks: Number of blocks per stack (default: 3)
            hidden_size: Hidden layer size in FC layers (default: 256)
            theta_size: Dimension of theta (basis expansion coefficients) (default: 32)
        """
        super()._setup(seq_len, pred_len, n_stacks, n_blocks, hidden_size, theta_size)
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
        input = self.extract_input(batch)
        output = self(input)
        return self.scaler.rescale(output, q1s_=q1s_, q2s_=q2s_, q3s_=q3s_)
