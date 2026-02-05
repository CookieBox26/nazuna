import torch
from abc import ABC, abstractmethod


class BaseScaler(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def scale(self, x, batch):
        pass

    @abstractmethod
    def rescale(self, z, batch):
        pass


class IqrScaler(BaseScaler):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.register_buffer('q1s', None)
        self.register_buffer('q2s', None)
        self.register_buffer('q3s', None)

    def save_quantiles(self, batch):
        """Save quantiles from a batch to use with mode='saved'."""
        q1s_ = batch.quantiles['full'][:, 0, :]
        q2s_ = batch.quantiles['full'][:, 1, :]
        q3s_ = batch.quantiles['full'][:, 2, :]
        self.q1s = q1s_[0:1, :]
        self.q2s = q2s_[0:1, :]
        self.q3s = q3s_[0:1, :]

    def _get_quantiles(self, batch):
        if self.mode == 'saved':
            return self.q1s, self.q2s, self.q3s
        self.save_quantiles(batch)
        q1s_ = batch.quantiles[self.mode][:, 0, :]
        q2s_ = batch.quantiles[self.mode][:, 1, :]
        q3s_ = batch.quantiles[self.mode][:, 2, :]
        return q1s_, q2s_, q3s_

    def scale(self, x, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        means = q2s_
        stds = q3s_ - q1s_
        return (x - means) / stds

    def rescale(self, z, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        means = q2s_
        stds = q3s_ - q1s_
        return z * stds + means
