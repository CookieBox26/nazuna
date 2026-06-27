import torch


class Adam(torch.optim.Adam):
    def __init__(self, *args, record_norms=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_norms = record_norms
        self._grad_sq_norms = []
        self._update_sq_norms = []

    @property
    def grad_norms(self):
        return torch.stack(self._grad_sq_norms).sqrt().cpu().numpy()

    @property
    def update_norms(self):
        return torch.stack(self._update_sq_norms).sqrt().cpu().numpy()

    @torch.no_grad()
    def step(self, closure=None):
        if self.record_norms:
            params = [p for group in self.param_groups for p in group['params']]
            device = params[0].device
            grad_sq_norm = torch.zeros((), device=device)
            for p in params:
                if p.grad is not None:
                    grad_sq_norm += p.grad.pow(2).sum()
            prev = [p.detach().clone() for p in params]
        loss = super().step(closure)
        if self.record_norms:
            update_sq_norm = torch.zeros((), device=device)
            for p, q in zip(params, prev):
                update_sq_norm += (p - q).pow(2).sum()
            self._grad_sq_norms.append(grad_sq_norm)
            self._update_sq_norms.append(update_sq_norm)
        return loss


class CoupledAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8):
        defaults = {'lr': lr, 'betas': betas, 'eps': eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = exp_avg / (1 - beta1 ** step)
                v_hat = exp_avg_sq / (1 - beta2 ** step)
                if p.ndim == 2:
                    v_hat = v_hat.mean(dim=0, keepdim=True).expand_as(p)
                p.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)
