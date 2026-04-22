from nazuna.workflow import run
import pytest


premise = '''
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
'''


tasks_0 = '''
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


template_train_with_baseline = '''
# =============== template ===============
[template]
template_type = "train_with_baseline"
criterion = "MSE"
baseline_model = "SimpleAverage"
model = "SimpleAverageVariableDecay"
data_range_train = [ 0.0, 0.8,]
data_range_eval = [ 0.8, 1.0,]
data_range_train_pilot = [ 0.0, 0.6,]
data_range_eval_pilot = [ 0.6, 0.8,]
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = 5
'''


template_train_with_baseline_multiseeds = '''
# =============== template ===============
[template]
template_type = "train_with_baseline_multiseeds"
criterion = "MSE"
baseline_model = "SimpleAverage"
model = "SimpleAverageVariableDecay"
data_range_train = [ 0.0, 0.8,]
data_range_eval = [ 0.8, 1.0,]
data_range_train_pilot = [ 0.0, 0.6,]
data_range_eval_pilot = [ 0.6, 0.8,]
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = 5
seeds = [ 0, 1, 2,]
'''


def validate_outputs(out_dir):
    assert out_dir.is_dir()
    assert (out_dir / 'config.toml').is_file()
    assert (out_dir / 'report.md').is_file()


def test_run_tasks_0(tmp_path):
    out_dir = tmp_path / 'tasks_0'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + tasks_0
    run(conf)
    validate_outputs(out_dir)


@pytest.mark.slow
def test_run_template_train_with_baseline(tmp_path):
    out_dir = tmp_path / 'template_train_with_baseline'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + \
        template_train_with_baseline
    run(conf)
    validate_outputs(out_dir)


@pytest.mark.slow
def test_run_template_train_with_baseline_multiseeds(tmp_path):
    out_dir = tmp_path / 'template_train_with_baseline_multiseeds'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + \
        template_train_with_baseline_multiseeds
    run(conf)
    validate_outputs(out_dir)
