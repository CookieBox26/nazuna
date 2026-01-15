import torch
from abc import ABC, abstractmethod


class BaseScaler(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def scale(self, x):
        pass

    @abstractmethod
    def rescale(self, z):
        pass


class StandardScaler(BaseScaler):
    def __init__(self, means_, stds_):
        super().__init__()
        self.register_buffer('means_', torch.tensor(means_, dtype=torch.float))
        self.register_buffer('stds_', torch.tensor(stds_, dtype=torch.float))
    def scale(self, x):
        return (x - self.means_) / self.stds_
    def rescale(self, z):
        return z * self.stds_ + self.means_


class IqrScaler(BaseScaler):
    def __init__(self, q1s_, q2s_, q3s_):
        super().__init__()
        self.register_buffer('means_', torch.tensor(q2s_, dtype=torch.float))
        self.register_buffer('stds_', torch.tensor(
            [q3 - q1 for q1, q3 in zip(q1s_, q3s_)],
            dtype=torch.float
        ))
    def scale(self, x):
        return (x - self.means_) / self.stds_
    def rescale(self, z):
        return z * self.stds_ + self.means_
