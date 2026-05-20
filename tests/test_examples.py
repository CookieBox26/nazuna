from nazuna.workflow import run
import nazuna.examples
import pytest
from pathlib import Path
from nazuna.utils import load_toml


@pytest.mark.parametrize(
    'identifier', [
        pytest.param('jma_train_dlinear', marks=pytest.mark.slow),
        pytest.param('jma_optuna_dlinear', marks=pytest.mark.slow),
    ],
)
def test_examples(identifier):
    conf_toml_path = nazuna.examples.get_conf_toml_path(identifier)
    out_dir = load_toml(conf_toml_path)['out_dir']
    Path(f'{out_dir}/report.md').unlink(missing_ok=True)
    run(conf_toml_path)
    assert Path(f'{out_dir}/report.md').is_file()
