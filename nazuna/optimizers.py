import numpy as np
import torch


class Adam(torch.optim.Adam):
    def __init__(self, *args, record_norms=False, record_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_norms = record_norms
        self.record_params = list(record_params) if record_params else []
        self._grad_sq_norms = []
        self._update_sq_norms = []
        self._exp_avg_sq_norms = []
        self._exp_avg_sq_sq_norms = []
        self._displacement_sq_norms = []
        self._init_params = None
        # Resolved (name, param) pairs to record per-parameter grad and
        # parameter norms for.
        self._individual_params = []
        self._individual_grad_sq_norms = {name: [] for name in self.record_params}
        self._individual_param_sq_norms = {name: [] for name in self.record_params}

    def bind_record_params(self, named_params):
        managed = {id(p) for group in self.param_groups for p in group['params']}
        for name in self.record_params:
            param = named_params.get(name)
            if param is not None and id(param) in managed:
                self._individual_params.append((name, param))

    @property
    def grad_norms(self):
        return torch.stack(self._grad_sq_norms).sqrt().cpu().numpy()

    @property
    def update_norms(self):
        return torch.stack(self._update_sq_norms).sqrt().cpu().numpy()

    @property
    def exp_avg_norms(self):
        return torch.stack(self._exp_avg_sq_norms).sqrt().cpu().numpy()

    @property
    def exp_avg_sq_norms(self):
        return torch.stack(self._exp_avg_sq_sq_norms).sqrt().cpu().numpy()

    @property
    def displacement_norms(self):
        return torch.stack(self._displacement_sq_norms).sqrt().cpu().numpy()

    def make_records(self):
        records = {'class_name': np.array(type(self).__name__)}
        if self.record_norms:
            records.update(
                grad_norms=self.grad_norms,
                update_norms=self.update_norms,
                exp_avg_norms=self.exp_avg_norms,
                exp_avg_sq_norms=self.exp_avg_sq_norms,
                displacement_norms=self.displacement_norms,
            )
        for name, sq_norms in self._individual_grad_sq_norms.items():
            if sq_norms:
                records[f'grad_norms/{name}'] = \
                    torch.stack(sq_norms).sqrt().cpu().numpy()
        for name, sq_norms in self._individual_param_sq_norms.items():
            if sq_norms:
                records[f'param_norms/{name}'] = \
                    torch.stack(sq_norms).sqrt().cpu().numpy()
        for name, param in self._individual_params:
            records[f'num_params/{name}'] = np.array(param.numel())
        return records

    def save_records(self, out_path):
        if not self.record_norms and not self._individual_params:
            return
        np.savez(out_path, **self.make_records())

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
            if self._init_params is None:
                self._init_params = [p.detach().clone() for p in params]
        for name, p in self._individual_params:
            grad_sq = (
                p.grad.pow(2).sum() if p.grad is not None
                else torch.zeros((), device=p.device)
            )
            self._individual_grad_sq_norms[name].append(grad_sq)
            self._individual_param_sq_norms[name].append(p.pow(2).sum())
        loss = super().step(closure)
        if self.record_norms:
            update_sq_norm = torch.zeros((), device=device)
            displacement_sq_norm = torch.zeros((), device=device)
            for p, q, p0 in zip(params, prev, self._init_params):
                update_sq_norm += (p - q).pow(2).sum()
                displacement_sq_norm += (p - p0).pow(2).sum()
            exp_avg_sq_norm = torch.zeros((), device=device)
            exp_avg_sq_sq_norm = torch.zeros((), device=device)
            for p in params:
                state = self.state[p]
                if 'exp_avg' in state:
                    exp_avg_sq_norm += state['exp_avg'].pow(2).sum()
                    exp_avg_sq_sq_norm += state['exp_avg_sq'].pow(2).sum()
            self._grad_sq_norms.append(grad_sq_norm)
            self._update_sq_norms.append(update_sq_norm)
            self._exp_avg_sq_norms.append(exp_avg_sq_norm)
            self._exp_avg_sq_sq_norms.append(exp_avg_sq_sq_norm)
            self._displacement_sq_norms.append(displacement_sq_norm)
        return loss


class FrozenOptimizer(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        return None


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
