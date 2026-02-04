from nazuna.optuna_helper import OptunaHelper


def test_merge_params():
    base_params = {'seq_len': 96, 'pred_len': 24, 'lr': 0.01}
    suggested = {'lr': 0.001, 'batch_size': 32}
    search_space = {'lr': ['log_uniform', 1e-5, 1e-2]}
    result = OptunaHelper.merge_params(base_params, suggested, search_space)
    assert result['seq_len'] == 96
    assert result['pred_len'] == 24
    assert result['lr'] == 0.001


def test_merge_params_with_none_base():
    suggested = {'lr': 0.001}
    search_space = {'lr': ['log_uniform', 1e-5, 1e-2]}
    result = OptunaHelper.merge_params(None, suggested, search_space)
    assert result['lr'] == 0.001


def test_merge_params_does_not_mutate_base():
    base_params = {'lr': 0.01}
    suggested = {'lr': 0.001}
    search_space = {'lr': ['log_uniform', 1e-5, 1e-2]}
    OptunaHelper.merge_params(base_params, suggested, search_space)
    assert base_params['lr'] == 0.01
