import pytest
from nazuna.examples import run_example


@pytest.mark.parametrize(
    'identifier', [
        pytest.param('jma_daily_eval_naive'),
        pytest.param('jma_daily_train_savd'),
        pytest.param('jma_daily_train_dlinear'),
        pytest.param('jma_daily_train_dlinearc'),
        pytest.param('jma_daily_train_patchtst'),
        pytest.param('jma_daily_train_residual'),
        pytest.param('jma_daily_optuna_savd'),
        pytest.param('jma_hourly_3m_diag'),
        pytest.param('jma_hourly_3m_eval_naive', marks=pytest.mark.slow),
        pytest.param('jma_hourly_36m_eval_naive', marks=pytest.mark.slow),
        pytest.param('jma_hourly_3m_train_savd', marks=pytest.mark.slow),
        pytest.param('jma_hourly_3m_train_dlinear', marks=pytest.mark.slow),
    ],
)
def test_examples(identifier):
    run_example(identifier)
