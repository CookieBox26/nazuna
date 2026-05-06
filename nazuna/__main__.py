from nazuna.workflow import run
import nazuna.examples
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str)
    parser.add_argument('-s', '--skip_task_ids', type=str, default='')
    parser.add_argument('-t', '--target_tasks', type=str, default='')
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args()

    conf = args.conf
    if args.example:
        conf = nazuna.examples.get_conf_toml_path(conf)
    run(conf, args.skip_task_ids, args.target_tasks)


if __name__ == '__main__':
    main()
