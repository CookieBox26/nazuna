import pytest
import torch
from nazuna.optimizers import CoupledAdam


def test_coupled_adam():
    lr, betas, eps = 0.1, (0.9, 0.999), 1e-8

    # 1D parameters bypass the coupling branch, so updates must match Adam.
    init_1d = torch.tensor([1.0, -2.0, 0.5])
    grads_1d = [
        torch.tensor([0.3, -0.1, 0.2]),
        torch.tensor([0.1, 0.4, -0.3]),
    ]
    p_coupled = torch.nn.Parameter(init_1d.clone())
    p_adam = torch.nn.Parameter(init_1d.clone())
    opt_coupled = CoupledAdam([p_coupled], lr=lr, betas=betas, eps=eps)
    opt_adam = torch.optim.Adam([p_adam], lr=lr, betas=betas, eps=eps)
    for grad in grads_1d:
        p_coupled.grad = grad.clone()
        p_adam.grad = grad.clone()
        opt_coupled.step()
        opt_adam.step()
    assert p_coupled.detach().tolist() == pytest.approx(p_adam.detach().tolist())

    # 2D parameters share v_hat averaged across dim 0.
    init_2d = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.2]])
    grads_2d = [
        torch.tensor([[0.3, -0.1], [0.2, 0.4], [-0.3, 0.1]]),
        torch.tensor([[0.1, 0.4], [-0.3, 0.2], [0.5, -0.2]]),
    ]
    p = torch.nn.Parameter(init_2d.clone())
    opt = CoupledAdam([p], lr=lr, betas=betas, eps=eps)

    beta1, beta2 = betas
    ref = init_2d.clone()
    exp_avg = torch.zeros_like(ref)
    exp_avg_sq = torch.zeros_like(ref)
    for step, grad in enumerate(grads_2d, start=1):
        p.grad = grad.clone()
        opt.step()
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
        m_hat = exp_avg / (1 - beta1 ** step)
        v_hat = exp_avg_sq / (1 - beta2 ** step)
        v_hat = v_hat.mean(dim=0, keepdim=True).expand_as(ref)
        ref = ref - lr * m_hat / (v_hat.sqrt() + eps)
    assert p.detach().flatten().tolist() == pytest.approx(ref.flatten().tolist())
