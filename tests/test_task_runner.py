from nazuna.data_manager import TimeSeriesDataManager
from nazuna.task_runner import TrainTaskRunner, EvalTaskRunner


def test_train_task_runner():
    path = 'nazuna/datasets/jma/weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv'
    conf_data = {
        'path': path, 'colname_timestamp': 'date', 'seq_len': 7 * 4, 'pred_len': 7,
        'white_list': ['temp_avg_nagoya', 'temp_avg_fukuoka'],
    }
    dm = TimeSeriesDataManager(**conf_data)
    assert len(dm.df) == 365

    conf_task = {
        'n_channel': 2, 'seq_len': 7 * 4, 'pred_len': 7,
    }
    runner = EvalTaskRunner(dm, conf_task)
    runner.run()

    runner = TrainTaskRunner(dm, conf_task)
