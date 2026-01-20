from nazuna.task_runner import run_tasks


conf_toml_str = '''
out_dir = "out/eval_jma_hourly"
exist_ok = true

[data]
path = [ "jma", "weather_japan_hourly_2025-09-01_2025-12-31_27_blocks.csv",]
colname_timestamp = "timestamp"
seq_len = 96  # 24 * 4
pred_len = 24

[[tasks]]
task_type = "eval"
data_range = [ 0.8, 1.0,]
criterion_cls_path = "nazuna.criteria.MAELoss"
criterion_params = { n_channel = 27, pred_len = 24 }
model_cls_path = "nazuna.models.simple_average.SimpleAverage"
model_params = { seq_len = 96, pred_len = 24, period_len = 24 }

[[tasks]]
task_type = "eval"
data_range = [ 0.8, 1.0,]
criterion_cls_path = "nazuna.criteria.MAELoss"
criterion_params = { n_channel = 27, pred_len = 24 }
model_cls_path = "nazuna.models.simple_average.SimpleAverage"
model_params = { seq_len = 72, pred_len = 24, period_len = 24 }
'''


def main():
    run_tasks(conf_toml_str)


if __name__ == '__main__':
    main()
