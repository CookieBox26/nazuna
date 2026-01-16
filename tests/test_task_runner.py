from nazuna.task_runner import EvalTaskRunner


def test_task_runner(get_data_manager):
    dm = get_data_manager()

    print()

    conf_task = {'n_channel': 2, 'seq_len': 7 * 4, 'pred_len': 7, 'period_len': 7}
    runner = EvalTaskRunner(dm, **conf_task)
    loss = runner.run()
    print('7 * 4 -> 7 : ', loss)

    conf_task = {'n_channel': 2, 'seq_len': 7 * 3, 'pred_len': 7, 'period_len': 7}
    runner = EvalTaskRunner(dm, **conf_task)
    loss = runner.run()
    print('7 * 3 -> 7 : ', loss)
