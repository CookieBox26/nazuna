from nazuna.task_runner import (
    EvalTaskRunner,
    Config,
    run_tasks,
)


def test_eval_task_runner(get_data_manager):
    dm = get_data_manager()
    out_dir = 'out/20260000-000000/task_0'
    conf_task = {'n_channel': 2, 'seq_len': 7 * 4, 'pred_len': 7, 'period_len': 7}

    runner = EvalTaskRunner(
        dm=dm, out_dir=out_dir,
        **conf_task,
    )
    loss = runner.run()

    assert runner.out_path.is_dir()
    runner.out_path.rmdir()


conf_toml_str = '''
out_dir = "out/20260000-000000"
exist_ok = true

[data]
path = "nazuna/datasets/jma/weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv"
colname_timestamp = "date"
seq_len = [7, 4]
pred_len = 7
white_list = ["temp_avg_nagoya", "temp_avg_fukuoka"]

[[tasks]]
task_type = "eval"
eval_range = [0.8, 1.0]

[[tasks]]
task_type = "eval"
eval_range = [0.8, 1.0]
'''


def test_run_tasks(tmp_path):
    toml_path = tmp_path / 'conf.toml'
    toml_path.write_text(conf_toml_str, newline='\n', encoding='utf8')
    run_tasks(Config.from_toml(toml_path))
    toml_path.unlink()
