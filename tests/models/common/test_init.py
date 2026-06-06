from nazuna.models.common import Patchifier
import torch


def test_patchifier(device):
    B, L, C = 1, 97, 1
    x = torch.arange(B * L * C, dtype=torch.float, device=device).reshape(B, L, C)

    patch_len, stride = 16, 8
    # patch  0:  0 -  15
    # patch  1:  8 -  23
    # ...
    # patch  9: 72 -  87
    # patch 10: 80 -  95
    # patch 11: 88 - 103

    patchifier = Patchifier(patch_len, stride, padding='end')
    patchifier.to(device)
    n_patches = patchifier.num_patches(L)
    assert n_patches == 12
    out = patchifier(x)
    assert out.shape == (B, C, n_patches, patch_len)
    expected = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(expected, device=device))
    expected = [88., 89., 90., 91., 92., 93., 94., 95., 96., 96., 96., 96., 96., 96., 96., 96.]
    torch.testing.assert_close(out[0, 0, 11], torch.tensor(expected, device=device))

    patchifier = Patchifier(patch_len, stride, padding='start')
    patchifier.to(device)
    n_patches = patchifier.num_patches(L)
    assert n_patches == 12
    out = patchifier(x)
    assert out.shape == (B, C, n_patches, patch_len)
    expected = [0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8.]
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(expected, device=device))
    expected = [81., 82., 83., 84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 96.]
    torch.testing.assert_close(out[0, 0, 11], torch.tensor(expected, device=device))

    patchifier = Patchifier(patch_len, stride, padding='')
    patchifier.to(device)
    n_patches = patchifier.num_patches(L)
    assert n_patches == 11
    out = patchifier(x)
    assert out.shape == (B, C, n_patches, patch_len)
    expected = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(expected, device=device))
    expected = [80., 81., 82., 83., 84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95.]
    torch.testing.assert_close(out[0, 0, 10], torch.tensor(expected, device=device))

    patch_len, stride = 16, 16
    # patch  0:  0 -  15
    # patch  1: 16 -  31
    # patch  2: 32 -  47
    # patch  3: 48 -  63
    # patch  4: 64 -  79
    # patch  5: 80 -  95
    # patch  6: 96 - 111

    patchifier = Patchifier(patch_len, stride, padding='end')
    patchifier.to(device)
    n_patches = patchifier.num_patches(L)
    assert n_patches == 7
    out = patchifier(x)
    assert out.shape == (B, C, n_patches, patch_len)
    expected = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(expected, device=device))
    expected = [96., 96., 96., 96., 96., 96., 96., 96., 96., 96., 96., 96., 96., 96., 96., 96.]
    torch.testing.assert_close(out[0, 0, 6], torch.tensor(expected, device=device))