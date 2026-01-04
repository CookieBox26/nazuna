import torch
from abc import ABC, abstractmethod


class BaseModel(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def create(cls, state_path=None, **kwargs):
        model = cls(**kwargs)
        if (state_path is not None) and (state_path != ''):
            model.load_state_dict(torch.load(state_path))
            model.eval()
        model.to(model.device)
        return model
