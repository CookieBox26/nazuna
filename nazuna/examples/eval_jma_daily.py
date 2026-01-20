from nazuna.data_manager import TimeSeriesDataManager
from nazuna.datasets import get_path
from nazuna.task_runner import EvalTaskRunner


def main():
    conf_data = {
        'path': get_path('jma', 'weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv'),
        'colname_timestamp': 'date', 'seq_len': 7 * 4, 'pred_len': 7,
        'white_list': ['temp_avg_nagoya', 'temp_avg_fukuoka'],
    }
    dm = TimeSeriesDataManager(**conf_data)
    conf_task = {'n_channel': 2, 'seq_len': 7 * 4, 'pred_len': 7, 'period_len': 7}
    runner = EvalTaskRunner(
        dm=dm, out_dir='out/eval_jma_daily/task_0', exist_ok=True,
        **conf_task,
    )
    loss = runner.run()
    print('7 * 4 -> 7 : ', loss)


if __name__ == '__main__':
    main()
