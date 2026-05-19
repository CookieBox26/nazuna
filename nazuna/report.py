from abc import ABC, abstractmethod
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for file output.
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import toml


class BasePlotter(ABC):
    @abstractmethod
    def _plot(self) -> str | Figure:
        pass

    def plot(self) -> str | None:
        with plt.rc_context({
            'svg.fonttype': 'path',  # convert text to paths for consistent rendering
            'svg.hashsalt': '',  # to make the IDs deterministic
            'font.size': 15,
            'lines.linewidth': 1.5,
        }):
            fig = self._plot()
            if isinstance(fig, str):
                return fig
            fig.savefig(self.graph_path, format='svg', bbox_inches='tight')
            plt.close(fig)
            return None

    def write_plot_section(self, f, report_path, alt_text, force_replot, cache) -> None:
        if not self.plot_data_exists:
            return
        err_msg = None
        entry = cache.get('graphs', {}).get(self.graph_path.name)
        is_up_to_date = (
            not force_replot and self.graph_path.exists() and entry is not None
            and entry.get('input_mtime_ns') == self.current_mtime
        )
        if not is_up_to_date:
            err_msg = self.plot()
            if err_msg is None:
                cache.setdefault('graphs', {})[self.graph_path.name] = {
                    'input_mtime_ns': self.current_mtime,
                }
            else:
                cache.get('graphs', {}).pop(self.graph_path.name, None)
        if err_msg is None:
            rel_path = self.graph_path.relative_to(report_path.parent)
            f.write(f'![{alt_text}]({rel_path.as_posix()})\n\n')
        else:
            f.write(f'{err_msg}\n\n')


class SamplePlotter(BasePlotter):
    """Plot sample data from DiagnosticsTaskRunner."""

    def __init__(self, graph_path: Path, sample_path: Path):
        self.graph_path = graph_path
        self.sample_path = sample_path
        self.plot_data_exists = sample_path.exists()
        self.current_mtime = (
            sample_path.stat().st_mtime_ns if self.plot_data_exists else None
        )

    def _plot(self) -> Figure:
        data = np.load(self.sample_path)
        values = data['values']
        columns = data['columns']
        timestamps = data['timestamps']
        timestamps = [str(t) for t in timestamps]
        if all(t.endswith(":00") for t in timestamps):
            timestamps = [t[:-3] for t in timestamps]

        fig, ax = plt.subplots(figsize=(5, 2))
        for i, col in enumerate(columns):
            ax.plot(timestamps, values[:, i], label=col, linewidth=2)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -1.3), ncol=2)
        tick_step = 12
        ax.set_xticks(range(0, len(timestamps), tick_step))
        ax.set_xticklabels(timestamps[::tick_step], rotation=90)
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)
        return fig


class PredPlotter(BasePlotter):
    """Plot prediction vs true (and baseline if available) for the first channel.

    The _plot method returns a message string when plotting is skipped,
    otherwise None.
    """

    def __init__(self, graph_path: Path, pred_path: Path):
        self.graph_path = graph_path
        self.pred_path = pred_path
        self.plot_data_exists = pred_path.exists()
        self.current_mtime = (
            pred_path.stat().st_mtime_ns if self.plot_data_exists else None
        )

    def _plot(self) -> str | Figure:
        npz = np.load(self.pred_path, allow_pickle=True)
        if 'seq_len' not in npz.files:
            return 'Skipped: model reference length (seq_len) is unknown.'
        seq_len = int(npz['seq_len'])
        data = npz['data'][-seq_len:, 0]
        data_future = npz['data_future'][:, 0]
        pred = npz['pred'][:, 0]
        has_baseline = 'baseline' in npz.files
        title = None
        if 'sample_index' in npz.files:
            sample_idx = int(npz['sample_index'])
            ts = str(npz['timestamp'])
            title = f'Eval sample {sample_idx} ({ts})'
        pred_len = len(pred)
        true_all = np.concatenate([data, data_future])
        x_true = range(len(true_all))
        x_pred = range(seq_len, seq_len + pred_len)

        fig, ax = plt.subplots(figsize=(8, 2))
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
        return fig


class TrainLossPlotter(BasePlotter):
    def __init__(self, graph_path: Path, history_path: Path):
        self.graph_path = graph_path
        self.history_path = history_path
        self.plot_data_exists = history_path.exists()
        self.current_mtime = (
            history_path.stat().st_mtime_ns if self.plot_data_exists else None
        )

    def _plot(self) -> Figure:
        history = toml.loads(self.history_path.read_text(encoding='utf8'))
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
        return fig


class SensitivityPlotter(BasePlotter):
    def __init__(self, graph_path: Path, history_path: Path):
        self.graph_path = graph_path
        self.history_path = history_path
        self.plot_data_exists = history_path.exists()
        self.current_mtime = (
            history_path.stat().st_mtime_ns if self.plot_data_exists else None
        )

    def _plot(self) -> Figure:
        history = toml.loads(self.history_path.read_text(encoding='utf8'))
        if not (
            'eval' in history['epochs'][0]
            and 'sensitivity' in history['epochs'][0]['eval']
        ):
            return 'No sensitivity data; skipping plot.'

        epochs = history['epochs']
        x = [e['i_epoch'] for e in epochs]
        sensitivity = [e['eval']['sensitivity'] for e in epochs]

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, sensitivity, linewidth=1.5)
        ax.set_xlabel('epoch')
        ax.set_ylabel('sensitivity')
        ax.grid(True, linestyle='--', linewidth=0.5)
        return fig


def _write_plot_section(f, report_path, task_runner, force_replot) -> None:
    cache = {}
    cache_path = task_runner.out_path / '.plot_cache.toml'
    if cache_path.exists():
        cache = toml.loads(cache_path.read_text(encoding='utf8'))

    SamplePlotter(
        graph_path=(task_runner.out_path / 'sample.npz'),
        sample_path=(task_runner.out_path / 'sample.svg'),
    ).write_plot_section(f, report_path, 'sample', force_replot, cache)

    TrainLossPlotter(
        graph_path=(task_runner.out_path / 'train_loss.svg'),
        history_path=(task_runner.out_path / 'train_loss_history.toml'),
    ).write_plot_section(f, report_path, 'train_loss', force_replot, cache)

    SensitivityPlotter(
        graph_path=(task_runner.out_path / 'sensitivity.svg'),
        history_path=(task_runner.out_path / 'train_loss_history.toml'),
    ).write_plot_section(f, report_path, 'sensitivity', force_replot, cache)

    for suffix in ['first', 'last']:
        PredPlotter(
            graph_path=(task_runner.out_path / f'pred_{suffix}.svg'),
            pred_path=(task_runner.out_path / f'pred_{suffix}.npz'),
        ).write_plot_section(f, report_path, f'pred_{suffix}', force_replot, cache)

    if cache.get('graphs'):
        cache_path.write_text(toml.dumps(cache), newline='\n', encoding='utf8')


def _write_report(
    f,
    report_path: Path,
    conf_toml_str: str,
    task_runners: list,
    suppress_plot: bool,
    force_replot: bool,
) -> None:
    f.write('### Configuration\n')
    f.write('```toml\n')
    f.write(conf_toml_str)
    f.write('```\n\n')

    f.write('### Result\n')
    for task_runner in task_runners:
        f.write(f'#### {task_runner.name}\n')
        if not task_runner.out_path.is_dir():
            f.write('No output path.\n\n')
            continue

        artifacts = [
            p.name for p in task_runner.out_path.iterdir() if (
                p.is_file()
                and p.name not in ('log.txt', 'result.toml', '.plot_cache.toml')
                and (p.suffix != '.svg')
            )
        ]
        if artifacts:
            f.write(f'Artifacts: {", ".join(artifacts)}\n\n')
        if not suppress_plot:
            _write_plot_section(f, report_path, task_runner, force_replot)

        result_path = task_runner.out_path / 'result.toml'
        if not result_path.is_file():
            continue
        f.write('```toml\n')
        f.write(toml.dumps(toml.loads(result_path.read_text(encoding='utf8'))))
        f.write('```\n')


def report(
    report_path: Path,
    conf_toml_str: str,
    task_runners: list,
    suppress_plot: bool = False,
    force_replot: bool = False,
) -> None:
    with report_path.open('w', newline='\n', encoding='utf8') as f:
        _write_report(
            f, report_path, conf_toml_str, task_runners,
            suppress_plot, force_replot,
        )
