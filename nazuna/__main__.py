from nazuna.workflow import run_tasks
import nazuna.examples
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str)
    parser.add_argument('--skip_task_ids', type=str, default='')
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args()

    conf = args.conf
    if args.example:
        conf = nazuna.examples.get_conf_toml_path(conf)

    run_tasks(
        conf,
        args.skip_task_ids,
    )


if __name__ == '__main__':
    main()
