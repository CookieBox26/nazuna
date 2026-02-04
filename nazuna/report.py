from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import toml


plt.rcParams['svg.fonttype'] = 'none'  # to reduce file size
plt.rcParams['svg.hashsalt'] = ''  # to make the IDs deterministic
plt.rcParams['font.size'] = 11


def _plot_sample(sample_path: Path, graph_path: Path) -> None:
    data = np.load(sample_path)
    values = data['values']
    columns = data['columns']
    timestamps = data['timestamps']

    timestamps = [str(t) for t in timestamps]
    if all(t.endswith(":00") for t in timestamps):
        timestamps = [t[:-3] for t in timestamps] 

    fig, ax = plt.subplots(figsize=(8, 3))
    for i, col in enumerate(columns):
        ax.plot(timestamps, values[:, i], label=col)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    tick_step = 4
    ax.set_xticks(range(0, len(timestamps), tick_step))
    ax.set_xticklabels(timestamps[::tick_step], rotation=90)
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)
    fig.savefig(graph_path, format='svg', bbox_inches='tight')
    plt.close(fig)


def report(
    report_path: Path,
    conf_toml_str: str,
    task_runners: list,
) -> None:
    with report_path.open('w', newline='\n', encoding='utf8') as f:
        f.write('### Configuration\n')
        f.write('```toml\n')
        f.write(conf_toml_str)
        f.write('```\n')
        f.write('\n')
        f.write('### Result\n')
        for task_runner in task_runners:
            f.write(f'#### {task_runner.name}\n')
            artifacts = [
                p.name for p in task_runner.out_path.iterdir()
                if p.is_file() and p.name != 'result.toml'
            ]
            if artifacts:
                f.write(f'Artifacts: {", ".join(artifacts)}\n\n')

            sample_path = task_runner.out_path / 'sample.npz'
            if sample_path.exists():
                graph_path = task_runner.out_path / 'graph.svg'
                _plot_sample(sample_path, graph_path)
                rel_path = graph_path.relative_to(report_path.parent)
                f.write(f'![graph]({rel_path.as_posix()})\n\n')

            f.write('```toml\n')
            f.write(toml.dumps(task_runner.result))
            f.write('```\n')
