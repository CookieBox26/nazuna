import pytest
from nazuna.examples import run_example


@pytest.mark.parametrize(
    'identifier', [
        pytest.param('jma_daily_eval_sa'),
        pytest.param('jma_daily_train_savd'),
        pytest.param('jma_hourly_3m_diag'),
        pytest.param('jma_hourly_3m_eval_sa', marks=pytest.mark.slow),
        pytest.param('jma_hourly_24m_eval_sa', marks=pytest.mark.slow),
        pytest.param('jma_hourly_3m_train_savd', marks=pytest.mark.slow),
        pytest.param('jma_hourly_3m_train_dlinear', marks=pytest.mark.slow),
    ],
)
def test_examples(identifier):
    run_example(identifier)
