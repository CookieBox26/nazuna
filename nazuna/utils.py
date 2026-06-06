from pathlib import Path
from importlib import import_module
import random
import numpy as np
import torch
import time
from contextlib import contextmanager
import datetime
import socket
import toml


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


def as_path_if_length_safe(s):
    """
    Returns a Path object if the given string, when interpreted as a path,
    satisfies the constraints: maximum path length <= 1000 characters
    and maximum component length <= 250 characters. Otherwise returns None.
    """
    if len(s) > 1000:
        return None
    p = Path(s)
    for part in p.parts:
        if len(part) > 250:
            return None
    return p


@contextmanager
def measure_time(info=None, raise_if_elapsed_over_min=-1):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        mins, secs = divmod(elapsed, 60)
        if info is not None:
            info['elapsed'] = f'{int(mins)} min {int(secs)} sec'
    if raise_if_elapsed_over_min > 0:
        if elapsed > raise_if_elapsed_over_min * 60:
            msg = f'Elapsed time exceeded {raise_if_elapsed_over_min} min:' \
                + f' {elapsed:.1f} sec'
            raise TimeoutError(msg)


def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def get_env_info():
    return {
        'hostname': socket.gethostname(),
        'torch_version': str(torch.__version__),
        'device_names': [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ],
    }


def load_toml(path: str | Path):
    text = Path(path).read_text(encoding='utf8')
    return toml.loads(text)
