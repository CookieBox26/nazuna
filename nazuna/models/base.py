import torch
from abc import ABC, abstractmethod


class BaseModel(torch.nn.Module, ABC):
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    @classmethod
    def create(cls, state_path=None, **kwargs):
        model = cls(**kwargs)
        if (state_path is not None) and (state_path != ''):
            model.load_state_dict(torch.load(state_path))
            model.eval()
        model.to(model.device)
        return model

    @abstractmethod
    def get_loss(self, batch, criterion):
        pass

    @abstractmethod
    def predict(self, batch):
        pass

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
