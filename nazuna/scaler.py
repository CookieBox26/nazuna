import torch
from abc import ABC, abstractmethod


class BaseScaler(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def scale(self, x, **kwargs):
        pass

    @abstractmethod
    def rescale(self, z, **kwargs):
        pass


class IqrScaler(BaseScaler):
    def __init__(self):
        super().__init__()

    def scale(self, x, q1s_, q2s_, q3s_):
        means = q2s_
        stds = q3s_ - q1s_
        return (x - means) / stds

    def rescale(self, z, q1s_, q2s_, q3s_):
        means = q2s_
        stds = q3s_ - q1s_
        return z * stds + means
