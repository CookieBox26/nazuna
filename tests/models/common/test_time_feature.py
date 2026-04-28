from nazuna.models.common import TimeFeatureEmbedding as TFE
import numpy as np
import torch
import pytest


def test_time_feature(device):
    TFE.Freq['hour']
    TFE.Freq['day']
    with pytest.raises(KeyError):
        TFE.Freq['ABC']

    tfe = TFE(device, freq='hour', d_model=3)
    assert isinstance(tfe.embed, torch.nn.Linear)
    assert tfe.embed.in_features == 4
    assert tfe.embed.out_features == 3
    assert tfe.embed.bias is None

    tsta = np.array([[
        '2026-01-01 00:00',
        '2026-01-01 01:00',
        '2026-01-01 02:00',
    ]])
    feats = tfe.get_feats(tsta)
    assert torch.allclose(feats, torch.tensor([[
        # to_hour_feat, to_dow_feat, to_day_feat, to_doy_feat
        [-0.5, 0., -0.5, -0.5],
        [1. / 23. - 0.5, 0., -0.5, -0.5],
        [2. / 23. - 0.5, 0., -0.5, -0.5],
    ]], device=device))
