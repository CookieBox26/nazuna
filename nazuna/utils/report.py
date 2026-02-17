from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import toml


plt.rcParams['svg.fonttype'] = 'path'  # convert text to paths for consistent rendering
plt.rcParams['svg.hashsalt'] = ''  # to make the IDs deterministic
plt.rcParams['font.size'] = 13


def _plot_sample(sample_path: Path, graph_path: Path) -> None:
    """Plot sample data from DiagnosticsTaskRunner."""
    data = np.load(sample_path)
    values = data['values']
    columns = data['columns']
    timestamps = data['timestamps']

    timestamps = [str(t) for t in timestamps]
    if all(t.endswith(":00") for t in timestamps):
        timestamps = [t[:-3] for t in timestamps]

    fig, ax = plt.subplots(figsize=(8, 2))
    for i, col in enumerate(columns):
        ax.plot(timestamps, values[:, i], label=col)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    tick_step = 4
    ax.set_xticks(range(0, len(timestamps), tick_step))
    ax.set_xticklabels(timestamps[::tick_step], rotation=90)
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)
    fig.savefig(graph_path, format='svg', bbox_inches='tight')
    plt.close(fig)


def _plot_pred(pred_path: Path, graph_path: Path) -> None:
    """Plot prediction vs true (and baseline if available) for the first channel."""
    npz = np.load(pred_path, allow_pickle=True)
    data = npz['data'][:, 0]
    data_future = npz['data_future'][:, 0]
    pred = npz['pred'][:, 0]
    has_baseline = 'baseline' in npz.files

    title = None
    if 'sample_index' in npz.files:
        sample_idx = int(npz['sample_index'])
        ts = str(npz['timestamp'])
        title = f'Eval sample {sample_idx} ({ts})'

    seq_len = len(data)
    pred_len = len(pred)
    true_all = np.concatenate([data, data_future])

    fig, ax = plt.subplots(figsize=(8, 2))
    x_true = range(len(true_all))
    x_pred = range(seq_len, seq_len + pred_len)
    ax.plot(x_true, true_all, label='true', color='black', linewidth=1)
    if has_baseline:
        baseline = npz['baseline'][:, 0]
        ax.plot(
            x_pred, baseline, label='baseline',
            color='tab:gray', linestyle='dashed', linewidth=2,
        )
    ax.plot(x_pred, pred, label='pred', color='tab:blue', linewidth=2)
    ax.axvline(x=seq_len - 1, color='tab:red', linewidth=1)

    if title is not None:
        ax.set_title(title, fontsize=13)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('step')
    ax.set_ylabel('y0')
    ax.grid(True, linestyle='--', linewidth=0.5)
    fig.savefig(graph_path, format='svg', bbox_inches='tight')
    plt.close(fig)


def _plot_train_loss(history_path: Path, graph_path: Path) -> None:
    history = toml.loads(history_path.read_text(encoding='utf8'))
    epochs = history['epochs']
    x = [e['i_epoch'] for e in epochs]
    train_loss = [e['train']['loss_per_sample'] for e in epochs]
    has_eval = 'eval' in epochs[0]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, train_loss, label='train', linewidth=1.5)
    if has_eval:
        eval_loss = [e['eval']['loss_per_sample'] for e in epochs]
        ax.plot(x, eval_loss, label='eval', linewidth=1.5)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss per sample')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, linestyle='--', linewidth=0.5)
    fig.savefig(graph_path, format='svg', bbox_inches='tight')
    plt.close(fig)


def report(
    report_path: Path,
    conf_toml_str: str,
    task_runners: list,
) -> None:
    with report_path.open('w', newline='\n', encoding='utf8') as f:
        f.write('### Configuration\n')
        f.write(f'```toml\n{conf_toml_str}```\n')
        f.write('\n')

        f.write('### Result\n')
        for task_runner in task_runners:
            f.write(f'#### {task_runner.name}\n')
            if not task_runner.out_path.is_dir():
                f.write('Not found\n\n')
                continue
            artifacts = [
                p.name for p in task_runner.out_path.iterdir()
                if p.is_file()
                and p.name != 'result.toml'
            ]
            if artifacts:
                f.write(f'Artifacts: {", ".join(artifacts)}\n\n')

            sample_path = task_runner.out_path / 'sample.npz'
            if sample_path.exists():
                graph_path = task_runner.out_path / 'graph.svg'
                _plot_sample(sample_path, graph_path)
                rel_path = graph_path.relative_to(report_path.parent)
                f.write(f'![graph]({rel_path.as_posix()})\n\n')

            history_path = task_runner.out_path / 'train_loss_history.toml'
            if history_path.exists():
                graph_path = task_runner.out_path / 'train_loss.svg'
                _plot_train_loss(history_path, graph_path)
                rel_path = graph_path.relative_to(report_path.parent)
                f.write(f'![train_loss]({rel_path.as_posix()})\n\n')

            pred_path = task_runner.out_path / 'pred_first.npz'
            if pred_path.exists():
                graph_path = task_runner.out_path / 'pred_first.svg'
                _plot_pred(pred_path, graph_path)
                rel_path = graph_path.relative_to(report_path.parent)
                f.write(f'![pred]({rel_path.as_posix()})\n\n')

            pred_last_path = task_runner.out_path / 'pred_last.npz'
            if pred_last_path.exists():
                graph_last_path = task_runner.out_path / 'pred_last.svg'
                _plot_pred(pred_last_path, graph_last_path)
                rel_path = graph_last_path.relative_to(report_path.parent)
                f.write(f'![pred_last]({rel_path.as_posix()})\n\n')

            result = toml.loads(
                (task_runner.out_path / 'result.toml').read_text(encoding='utf8'),
            )
            f.write('```toml\n')
            f.write(toml.dumps(result))
            f.write('```\n')
