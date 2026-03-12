from nazuna.utils import as_path_if_length_safe
from pathlib import Path
import pytest


@pytest.mark.parametrize(
    ['s', 'expected'],
    [
        pytest.param('a' * 200, True, id='00'),
        pytest.param('a' * 300, False, id='01'),
        pytest.param(('/' + 'a' * 200) * 4, True, id='10'),
        pytest.param(('/' + 'a' * 200) * 5, False, id='11'),
    ],
)
def test_as_path_if_length_safe(s, expected):
    p = as_path_if_length_safe(s)
    if expected:
        assert isinstance(p, Path)
    else:
        assert p is None
