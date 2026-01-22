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
