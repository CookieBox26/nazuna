"""
Ex. python -m nazuna.examples eval_sa_jma_daily
"""
from nazuna.examples import run_example
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('identifier', type=str)
    args = parser.parse_args()
    run_example(args.identifier)


if __name__ == '__main__':
    main()
