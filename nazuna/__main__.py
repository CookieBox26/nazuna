from nazuna.task_runner import run_tasks
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to a TOML config file')
    parser.add_argument('-s', '--skip_task_ids', type=str, default='')
    args = parser.parse_args()

    run_tasks(
        args.config,
        args.skip_task_ids,
    )


if __name__ == '__main__':
    main()
