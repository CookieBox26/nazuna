from importlib.resources import files


def get_dataset_path(category: str, filename: str) -> str:
    return str(files(f'nazuna.datasets.{category}').joinpath(filename))
