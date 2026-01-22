"""
Time series prediction model using periodic sin/cos features.

Takes timesteps as input, generates sin/cos waveforms with multiple periods as features,
and outputs predictions through linear transformation.
"""

from abc import abstractmethod
from nazuna.models.base import BaseModel
import torch
import torch.nn as nn
import numpy as np


class CircularHelper:
    """
    Helper class that caches and efficiently provides periodic sin/cos waveforms.
    """

    @staticmethod
    def get_sin(n, m):
        y = np.sin(2.0 * np.pi * (float(m) / float(n)))
        return 0.0 if (np.abs(y) < 1e-15) else y

    @staticmethod
    def get_cos(n, m):
        y = np.cos(2.0 * np.pi * (float(m) / float(n)))
        return 0.0 if (np.abs(y) < 1e-15) else y

    @staticmethod
    def tile(v, offset, target_len):
        v_ret = v.copy()
        v_ret = v_ret[offset:]
        while len(v_ret) < target_len:
            v_ret += v
        return torch.tensor(np.array(v_ret[:target_len]), dtype=torch.float32)

    def __init__(self, pred_len, periods, device):
        self.periods = periods
        self.device = device

        cache_sin_raw = {}
        cache_cos_raw = {}
        for n in self.periods:
            cache_sin_raw[n] = []
            cache_cos_raw[n] = []
            for m in range(n):
                cache_sin_raw[n].append(CircularHelper.get_sin(n, m))
                cache_cos_raw[n].append(CircularHelper.get_cos(n, m))

        self.cache_sin = {}
        self.cache_cos = {}
        for n in self.periods:
            self.cache_sin[n] = {}
            self.cache_cos[n] = {}
            for m in range(n):
                self.cache_sin[n][m] = CircularHelper.tile(cache_sin_raw[n], m, pred_len).to(device)
                self.cache_cos[n][m] = CircularHelper.tile(cache_cos_raw[n], m, pred_len).to(device)

        self.len_features = 2 * len(self.periods)
        if 2 in self.periods:
            self.len_features -= 1

    def get_wave(self, x):
        waves = []
        for n in self.periods:
            v = x[:, :1].fmod(n)
            if n != 2:
                waves.append(torch.stack(
                    [self.cache_sin[n][int(v_.item())].clone().detach() for v_ in v]
                ))
            waves.append(torch.stack(
                [self.cache_cos[n][int(v_.item())].clone().detach() for v_ in v]
            ))
        return torch.stack(waves)


class BaseCircular(BaseModel):
    """
    Base model for time series prediction using periodic sin/cos features.
    """

    def _setup(self, pred_len, n_channel, periods):
        self.pred_len = pred_len
        self.n_channel = n_channel
        self.periods = periods

    def _init_helper_and_linear(self, device):
        self.helper = CircularHelper(self.pred_len, self.periods, device)
        self.linear = nn.Linear(
            in_features=self.helper.len_features, out_features=self.n_channel
        )

    def forward(self, tste_future):
        """
        Forward pass.

        Args:
            tste_future (Tensor): Future timesteps, shape [batch, pred_len]

        Returns:
            Tensor: Predictions, shape [batch, pred_len, n_channel]
        """
        v = self.helper.get_wave(tste_future)  # [len_features, batch, pred_len]
        v = v.permute(1, 2, 0)  # [batch, pred_len, len_features]
        v = self.linear(v)  # [batch, pred_len, n_channel]
        return v

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


class Circular(BaseCircular):
    def _setup(self, pred_len: int, n_channel: int, periods: list[int]) -> None:
        """
        Args:
            pred_len: Prediction length
            n_channel: Number of output channels
            periods: List of periods to use (default: [2, 3, ..., 24])
        """
        super()._setup(pred_len, n_channel, periods)

    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        self._init_helper_and_linear(device)
        self.to(device)

    def extract_input(self, batch):
        """
        Extract input (future timesteps) from batch.

        The Circular model uses future timesteps as input, not past data.
        """
        return batch.tste_future[:, :self.pred_len]

    def extract_target(self, batch):
        """
        Extract target values from batch.
        """
        return batch.data_future[:, :self.pred_len]

    def predict(self, batch):
        """
        Run prediction and return output.
        """
        input_ = self.extract_input(batch)
        return self(input_)
