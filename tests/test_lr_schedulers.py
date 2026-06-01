import pytest
import torch
from nazuna.lr_schedulers import WarmupInverseSqrtLR


def test_warmup_inverse_sqrt_lr():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = WarmupInverseSqrtLR(
        optimizer, warmup_steps=4, start_factor=0.1, decay_power=0.5,
    )

    lrs = [scheduler.get_last_lr()[0]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    expected = [0.325, 0.55, 0.775, 1.0, (4 / 5) ** 0.5, (4 / 6) ** 0.5, (4 / 7) ** 0.5]
    assert lrs == pytest.approx(expected)
