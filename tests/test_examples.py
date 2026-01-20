import pytest
import nazuna.examples.eval_jma_daily
import nazuna.examples.eval_jma_hourly


def test_examples_eval_jma_daily():
    nazuna.examples.eval_jma_daily.main()

@pytest.mark.slow
def test_examples_eval_jma_hourly():
    nazuna.examples.eval_jma_hourly.main()
