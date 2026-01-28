"""
Ex. python -m nazuna ./my_config.toml
"""
from nazuna.task_runner import run_tasks
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to a TOML config file')
    args = parser.parse_args()
    run_tasks(args.config)


if __name__ == '__main__':
    main()
