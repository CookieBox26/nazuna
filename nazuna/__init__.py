from importlib import import_module


def load_class(path):
    try:
        module_path, class_name = path.rsplit('.', 1)
        module = import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError):
        raise ImportError(path)
    return cls
