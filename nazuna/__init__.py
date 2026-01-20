from importlib import import_module
import random
import numpy as np
import torch
import time
from contextlib import contextmanager


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def load_class(path):
    try:
        module_path, class_name = path.rsplit('.', 1)
        module = import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError):
        raise ImportError(path)
    return cls


@contextmanager
def measure_time(info):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        mins, secs = divmod(elapsed, 60)
        info['elapsed'] = f'{int(mins)} min {int(secs)} sec'
