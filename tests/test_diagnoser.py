import numpy as np
import pytest
from nazuna.utils.diagnoser import Diagnoser


@pytest.fixture(scope='module')
def dm(get_data_manager):
    return get_data_manager({
        'seq_len': 7 * 4, 'pred_len': 7,
        'white_list': ['temp_avg_nagoya', 'temp_avg_fukuoka'],
    })


@pytest.fixture(scope='module')
def diagnoser(dm):
    return Diagnoser(dm, data_range=(0.0, 1.0))


def test_measure_seasonality(diagnoser, dm):
    result = diagnoser.measure_seasonality(period=7)

    expected_keys = {
        'cols_org', 'seasonality_per_channel', 'seasonality_mean',
        'period', 'trend_var_per_channel', 'seasonal_var_per_channel',
        'residual_var_per_channel',
    }
    assert set(result.keys()) == expected_keys
    assert result['period'] == 7

    for val in result['seasonality_per_channel'].values():
        assert 0.0 <= val <= 1.0

    values = list(result['seasonality_per_channel'].values())
    expected_mean = float(np.mean(values))
    assert abs(result['seasonality_mean'] - expected_mean) < 1e-10

    for col, org_name in result['cols_org'].items():
        i_col = list(diagnoser.df.columns).index(col)
        assert org_name == dm.cols_org[i_col]

    cols = set(result['seasonality_per_channel'].keys())
    assert set(result['trend_var_per_channel'].keys()) == cols
    assert set(result['seasonal_var_per_channel'].keys()) == cols
    assert set(result['residual_var_per_channel'].keys()) == cols

    for col in result['seasonality_per_channel']:
        assert result['trend_var_per_channel'][col] >= 0.0
        assert result['seasonal_var_per_channel'][col] >= 0.0
        assert result['residual_var_per_channel'][col] >= 0.0


def test_measure_seasonality_short_series_returns_nan(get_data_manager):
    dm_short = get_data_manager({
        'seq_len': 10, 'pred_len': 3,
        'white_list': ['temp_avg_nagoya'],
    })
    # data_range=(0.0, 0.01) yields ~15 rows; period=14 requires 28 rows,
    # so len(series) < 2*period is satisfied.
    diag = Diagnoser(dm_short, data_range=(0.0, 0.01))
    result = diag.measure_seasonality(period=14)
    for val in result['seasonality_per_channel'].values():
        assert np.isnan(val)
    assert np.isnan(result['seasonality_mean'])


def test_measure_seasonality_constant_series(dm):
    diag = Diagnoser(dm, data_range=(0.0, 1.0))
    # Replace data with constants; STL may produce small numerical residuals,
    # so we check that seasonality is near zero rather than exactly zero.
    for col in diag.df.columns:
        diag.df[col] = 42.0
    result = diag.measure_seasonality(period=7)
    for val in result['seasonality_per_channel'].values():
        assert val < 0.1
    assert result['seasonality_mean'] < 0.1


def test_measure_quartiles(diagnoser, dm):
    result = diagnoser.measure_quartiles()

    expected_keys = {
        'cols_org', 'q1_per_channel', 'q2_per_channel', 'q3_per_channel',
    }
    assert set(result.keys()) == expected_keys

    cols = set(diagnoser.df.columns)
    assert set(result['q1_per_channel'].keys()) == cols
    assert set(result['q2_per_channel'].keys()) == cols
    assert set(result['q3_per_channel'].keys()) == cols

    for col in cols:
        assert result['q1_per_channel'][col] <= result['q2_per_channel'][col]
        assert result['q2_per_channel'][col] <= result['q3_per_channel'][col]

    for col, org_name in result['cols_org'].items():
        i_col = list(diagnoser.df.columns).index(col)
        assert org_name == dm.cols_org[i_col]

    # Verify against numpy directly
    for col in cols:
        series = diagnoser.df[col].values
        q1, q2, q3 = np.percentile(series, [25, 50, 75])
        assert abs(result['q1_per_channel'][col] - q1) < 1e-10
        assert abs(result['q2_per_channel'][col] - q2) < 1e-10
        assert abs(result['q3_per_channel'][col] - q3) < 1e-10


def test_sample(diagnoser):
    result = diagnoser.sample()
    assert 'values' in result
    assert 'columns' in result
    assert 'timestamps' in result
    assert result['values'].shape[1] <= 4
    assert result['values'].shape[0] <= 96

    # Requesting more channels than available should clamp to actual count
    result_many_ch = diagnoser.sample(n_channels=10)
    assert len(result_many_ch['columns']) == 2
    assert result_many_ch['values'].shape[1] == 2

    # Requesting more steps than available should clamp to data length
    n_rows = len(diagnoser.df)
    result_many_steps = diagnoser.sample(n_steps=n_rows + 100)
    assert result_many_steps['values'].shape[0] == n_rows

    # Custom params
    result_custom = diagnoser.sample(n_channels=1, n_steps=10)
    assert result_custom['values'].shape == (10, 1)
    assert len(result_custom['columns']) == 1
    assert len(result_custom['timestamps']) == 10

    # Returned columns should match df columns in order
    result_2ch = diagnoser.sample(n_channels=2)
    assert result_2ch['columns'] == list(diagnoser.df.columns[:2])


def test_run(diagnoser):
    result, data = diagnoser.run(period=7)
    assert isinstance(result, dict)
    assert isinstance(data, dict)

    assert 'seasonality' in result
    assert 'seasonality_per_channel' in result['seasonality']
    assert 'quartiles' in result
    assert 'q1_per_channel' in result['quartiles']

    assert 'values' in data
    assert 'columns' in data
    assert 'timestamps' in data
