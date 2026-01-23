import pytest
from nazuna.examples import run_example


@pytest.mark.parametrize(
    'identifier', [
        pytest.param('eval_sa_jma_daily'),
        pytest.param('eval_sa_jma_hourly_3m', marks=pytest.mark.slow),
        pytest.param('eval_sa_jma_hourly_24m', marks=pytest.mark.slow),
        pytest.param('train_savd_jma_daily'),
        pytest.param('train_savd_jma_hourly_3m', marks=pytest.mark.slow),
        pytest.param('train_savdc_jma_daily'),
        pytest.param('train_dlinear_jma_hourly_3m', marks=pytest.mark.slow),
    ],
)
def test_examples(identifier):
    run_example(identifier)
