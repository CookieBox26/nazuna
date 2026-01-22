from nazuna.task_runner import run_tasks


conf_toml_str = '''
out_dir = "out/eval_jma_daily"
exist_ok = true

[data]
path = [ "jma", "weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv",]
colname_timestamp = "date"
seq_len = 28
pred_len = 7
white_list = [ "temp_avg_nagoya", "temp_avg_fukuoka",]

[[tasks]]
task_type = "eval"
data_range_eval = [ 0.8, 1.0,]
criterion_cls_path = "nazuna.criteria.MAELoss"
criterion_params = { n_channel = 2, pred_len = 7 }
model_cls_path = "nazuna.models.simple_average.SimpleAverage"
model_params = { seq_len = 28, pred_len = 7, period_len = 7 }
'''


def main():
    run_tasks(conf_toml_str)


if __name__ == '__main__':
    main()
