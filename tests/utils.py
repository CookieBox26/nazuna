from nazuna.task_runners import EvalTaskRunner
from nazuna.tools import parse_param_conf
import toml


def set_training(model, training: bool):
    if training:
        model.train()
    else:
        model.eval()


def create_from_doc(model_cls, device):
    text = parse_param_conf(model_cls)
    d = toml.loads(text)['definitions'][model_cls.__name__]
    cls_, params_ = EvalTaskRunner.extract_model_config(d)
    assert cls_ is model_cls
    model = cls_(device, **params_)
    assert isinstance(model, model_cls)
    return model
