from nazuna.models.common import TimeFeatureEmbedding as TFE
import numpy as np
import torch
import pytest


def test_time_feature():
    TFE.Freq['Hour']
    TFE.Freq['Day']
    with pytest.raises(KeyError):
        TFE.Freq['ABC']

    tfe = TFE(d_model=3, freq='Hour')
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
    assert np.allclose(feats, np.array([[
        # to_hour_feat, to_dow_feat, to_day_feat, to_doy_feat
        [-0.5, 0., -0.5, -0.5],
        [1. / 23. - 0.5, 0., -0.5, -0.5],
        [2. / 23. - 0.5, 0., -0.5, -0.5],
    ]]))
