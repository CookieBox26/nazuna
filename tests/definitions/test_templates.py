from importlib.resources import files
from nazuna.definitions import get_definitions
from nazuna.models._base import BasicBaseModel
from nazuna.task_runners import EvalTaskRunner
from nazuna.utils import load_class
import pytest


template_data = {
    'seq_len_raw': -1,
    'seq_len': 96,
    'pred_len': 24,
    'n_channel': 7,
    'period_len': 24,
    'seq_len_sample': 97,
    'pred_len_sample': 24,
}


def collect_model_definitions():
    cases = []
    for p in sorted(files('nazuna.definitions').iterdir()):
        if not p.name.endswith('.toml.j2'):
            continue
        stem = p.name[:-len('.toml.j2')]
        defs = get_definitions(stem, dict(template_data))['definitions']
        for name, conf in defs.items():
            if not isinstance(conf, dict) or 'cls_path' not in conf:
                continue
            if issubclass(load_class(conf['cls_path']), BasicBaseModel):
                cases.append(pytest.param(conf, id=f'{stem}::{name}'))
    return cases


@pytest.mark.parametrize('conf', collect_model_definitions())
def test_can_instantiate(device, conf):
    cls_, params_ = EvalTaskRunner.extract_model_config(conf)
    model = cls_(device, **params_)
    assert isinstance(model, cls_)
