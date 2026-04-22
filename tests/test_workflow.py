from nazuna.workflow import run


conf_toml_str_0 = '''
exist_ok = true

# =============== data ===============
[data]
path = [ "jma", "weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv",]
colname_timestamp = "date"
seq_len = 28
pred_len = 7
white_list = [ "temp_avg_nagoya", "temp_avg_fukuoka",]

# =============== definitions ===============
[definitions.MSE]
cls_path = "nazuna.criteria.MSE"
params = { n_channel = 2, pred_len = 7 }

[definitions.SimpleAverage]
cls_path = "nazuna.models.simple_average.SimpleAverage"
params = { seq_len = 28, pred_len = 7, period_len = 7 }

[definitions.SimpleAverageVariableDecay]
cls_path = "nazuna.models.simple_average.SimpleAverageVariableDecay"
params = { seq_len = 28, pred_len = 7, period_len = 7 }

[definitions.BatchSamplerShuffle]
cls_path = "nazuna.batch_samplers.BatchSamplerShuffle"
params = { batch_size = 32 }

[definitions.Adam]
cls_path = "torch.optim.Adam"
params = { lr = 0.001 }

[definitions.CosineAnnealingLR]
cls_path = "torch.optim.lr_scheduler.CosineAnnealingLR"
params = { T_max = 5 }

# =============== tasks ===============
[[tasks]]
task_type = "eval"
name = "Eval Baseline"
data_range_eval = [ 0.8, 1.0,]
criterion = "MSE"
model = "SimpleAverage"

[[tasks]]
task_type = "train"
name = "Pilot"
data_range_train = [ 0.0, 0.6,]
data_range_eval = [ 0.6, 0.8,]
criterion = "MSE"
model = "SimpleAverageVariableDecay"
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = 5
early_stop = true

[[tasks]]
task_type = "train"
name = "Train"
data_range_train = [ 0.0, 0.8,]
criterion = "MSE"
model = "SimpleAverageVariableDecay"
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = { task_name = "Pilot" }
'''


def test_run_0(tmp_path):
    out_dir = tmp_path / 'workflow_0'
    run(f'out_dir = "{out_dir.as_posix()}"\n' + conf_toml_str_0)
    assert out_dir.is_dir()
