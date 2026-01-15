import torch
from abc import ABC, abstractmethod


class BaseModel(torch.nn.Module, ABC):
    """
    Base class for time-series forecasting models.
    Subclasses must implement _init_layers() to define their layers.
    The device is handled by this base class, so subclasses don't need to manage it.
    """
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device
        self._init_layers(**kwargs)
        self.to(device)

    @abstractmethod
    def _init_layers(self, **kwargs):
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
        pass

    @abstractmethod
    def predict(self, batch):
        pass

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
