import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Inspector:

    @staticmethod
    @torch.no_grad()
    def inspect(
        model,
        criterion,
        batches,
        delta_scale=1e-3,
        num_directions=8,
    ):
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError('No trainable parameters found.')

        w0 = parameters_to_vector(params).detach().clone()

        loss_0 = 0.0
        n_sample = 0
        for batch in batches:
            pred, _ = model.predict(batch)
            true = model.extract_true(batch)
            loss = criterion(pred, true)
            loss_0 += loss.batch_sum()
            n_sample += loss.batch_size
        loss_0 /= n_sample

        max_ratio = 0.0
        for _ in range(num_directions):
            delta = torch.randn_like(w0)
            delta = delta / delta.norm() * delta_scale
            vector_to_parameters(w0 + delta, params)

            loss_d = 0.0
            n_sample_d = 0
            for batch in batches:
                pred, _ = model.predict(batch)
                true = model.extract_true(batch)
                loss = criterion(pred, true)
                loss_d += loss.batch_sum()
                n_sample_d += loss.batch_size
            loss_d /= n_sample_d

            ratio = abs(loss_d - loss_0) / delta_scale
            if ratio > max_ratio:
                max_ratio = ratio

        vector_to_parameters(w0, params)
        return {'sensitivity': max_ratio}
