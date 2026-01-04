import pytest


@pytest.mark.parametrize('a, b, expected', [
    [1, 1, 2],
    [1, 2, 3],
])
def test_sample(a, b, expected):
    assert a + b == expected
