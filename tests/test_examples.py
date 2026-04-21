from nazuna.task_runners import run_tasks
import nazuna.examples
import pytest


@pytest.mark.parametrize(
    'identifier', [
        pytest.param('jma_train_dlinear', marks=pytest.mark.slow),
        pytest.param('jma_optuna_dlinear', marks=pytest.mark.slow),
    ],
)
def test_examples(identifier):
    conf_toml_path = nazuna.examples.get_conf_toml_path(identifier)
    run_tasks(conf_toml_path)
