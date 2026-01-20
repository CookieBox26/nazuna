from nazuna.data_manager import TimeSeriesDataManager
from nazuna.datasets import get_path
from nazuna.task_runner import EvalTaskRunner


def main():
    conf_data = {
        'path': get_path('jma', 'weather_japan_hourly_2025-09-01_2025-12-31_27_blocks.csv'),
        'colname_timestamp': 'timestamp', 'seq_len': 24 * 4, 'pred_len': 24,
    }
    dm = TimeSeriesDataManager(**conf_data)

    print()

    conf_task = {'n_channel': 27, 'seq_len': 24 * 4, 'pred_len': 24}
    runner = EvalTaskRunner(
        dm=dm, out_dir='out/eval_jma_hourly/task_0', exist_ok=True,
        **conf_task,
    )
    loss = runner.run()
    print('24 * 4 -> 24 : ', loss)

    conf_task = {'n_channel': 27, 'seq_len': 24 * 3, 'pred_len': 24}
    runner = EvalTaskRunner(
        dm=dm, out_dir='out/eval_jma_hourly/task_1', exist_ok=True,
        **conf_task,
    )
    loss = runner.run()
    print('24 * 3 -> 24 : ', loss)


if __name__ == '__main__':
    main()
