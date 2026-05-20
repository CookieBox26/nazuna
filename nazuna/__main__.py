from nazuna.workflow import run
import nazuna.examples
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str)
    parser.add_argument('-s', '--skip_task_ids', type=str, default='')
    parser.add_argument('-t', '--target_tasks', type=str, default='')
    parser.add_argument('-f', '--force_rerun', action='store_true')
    parser.add_argument('--suppress_plot', action='store_true')
    parser.add_argument('--force_replot', action='store_true')
    parser.add_argument('--report_only', action='store_true')
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args()

    conf = args.conf
    if args.example:
        conf = nazuna.examples.get_conf_toml_path(conf)
    run(
        conf, args.skip_task_ids, args.target_tasks, args.force_rerun,
        args.suppress_plot, args.force_replot, args.report_only,
    )


if __name__ == '__main__':
    main()
