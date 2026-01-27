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

    def _get_quantiles(self, batch):
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
