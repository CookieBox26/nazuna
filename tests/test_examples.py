from nazuna.workflow import run
import nazuna.examples
from nazuna.tools import WorkflowResult
import pytest


@pytest.mark.parametrize(
    'identifier', [
        pytest.param('jma_train_dlinear', marks=pytest.mark.slow),
        pytest.param('jma_optuna_dlinear', marks=pytest.mark.slow),
    ],
)
def test_examples(identifier):
    conf_toml_path = nazuna.examples.get_conf_toml_path(identifier)
    run(conf_toml_path)
    wr = WorkflowResult.from_conf_toml_path(conf_toml_path)
    for k, v in wr.out_paths.items():
        print(k, v)