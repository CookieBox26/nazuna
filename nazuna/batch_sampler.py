from abc import ABC
import numpy as np
import inspect


class BatchSampler:
    """A sampler that splits samples into batches in sequential order.
    """

    @classmethod
    def create(cls, n_sample, batch_size, **kwargs):
        if 'argnames' not in cls.__dict__:
            cls.argnames = set(inspect.signature(cls.__init__).parameters.keys())
            cls.argnames -= {'self', 'n_sample', 'batch_size'}
        filtered = {k: v for k, v in kwargs.items() if k in cls.argnames}
        return cls(n_sample, batch_size, **filtered)

    def __init__(self, n_sample: int, batch_size: int) -> None:
        """
        Args:
            n_sample: Total number of samples.
            batch_size: Batch size.
        """
        self.n_sample = n_sample
        self.batch_size = batch_size
        self.n_batch = int(np.ceil(self.n_sample / batch_size))

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        self.i_batch_actual = -1
        return self

    def _get_i_batch(self, i_batch):
        # Returns the list of sample indices belonging to the i-th batch
        # when samples are split sequentially into batches.
        # The caller must ensure that i_batch <= n_batch - 1.
        list_indices = [i_batch * self.batch_size + i for i in range(self.batch_size)]
        if i_batch == self.n_batch - 1:
            list_indices = [i for i in list_indices if i <= self.n_sample - 1]
        return list_indices

    def __next__(self):
        self.i_batch_actual += 1
        if self.i_batch_actual == self.n_batch:
            raise StopIteration()
        return self._get_i_batch(self.i_batch_actual)


class BatchSamplerRandom(BatchSampler, ABC):
    def __init__(self, n_sample, batch_size, seed=0):
        super().__init__(n_sample, batch_size)
        self.rng = np.random.default_rng(seed)


class BatchSamplerShuffle(BatchSamplerRandom):
    """A sampler that shuffles samples and splits them into batches.

    - At the start of each iteration, sample indices are shuffled,
      and batches are created based on the shuffled order.
    """

    def __init__(self, n_sample: int, batch_size: int, seed: int = 0) -> None:
        """
        Args:
            n_sample: Total number of samples.
            batch_size: Batch size.
            seed: Random seed.
        """
        super().__init__(n_sample, batch_size, seed)
        self.sample_ids_shuffled = [i for i in range(self.n_sample)]

    def __iter__(self):
        # Prepare a shuffled list of sample indices.
        self.sample_ids_shuffled.sort()
        self.rng.shuffle(self.sample_ids_shuffled)
        return super().__iter__()

    def __next__(self):
        # Get the base sampler output and map it to the shuffled sample indices.
        list_indices = super().__next__()
        return [self.sample_ids_shuffled[i] for i in list_indices]


class BatchSamplerBatchShuffle(BatchSamplerRandom):
    # A batch sampler that shuffles only the batch order.
    # Samples within each batch remain consecutive.
    def __init__(self, n_sample, batch_size, seed=0):
        super().__init__(n_sample, batch_size, seed)
        self.batch_ids_shuffled = [i for i in range(self.n_batch)]

    def __iter__(self):
        # Prepare a shuffled list of batch indices.
        self.batch_ids_shuffled.sort()
        self.rng.shuffle(self.batch_ids_shuffled)
        return super().__iter__()

    def __next__(self):
        self.i_batch_actual += 1
        if self.i_batch_actual == self.n_batch:
            raise StopIteration()
         # Map the current batch index to the shuffled batch order.
        return self._get_i_batch(self.batch_ids_shuffled[self.i_batch_actual])


class BatchSamplerPeriodic(BatchSamplerRandom):
    # A batch sampler that allows only indices congruent modulo `period` to be grouped into the same batch.
    # For example, for daily data with period=7, each batch contains samples from the same day of the week.
    # Currently, consecutive samples of the same weekday are grouped together.
    def __init__(self, n_sample, batch_size, seed=0, period=7):
        self.period = period
        super().__init__(n_sample, batch_size, seed)
        # Group all sample indices by their remainder modulo `period`.
        self.sample_ids_grouped = [[] for r in range(self.period)]
        for i_sample in range(self.n_sample):
            self.sample_ids_grouped[i_sample % self.period].append(i_sample)
        # Recompute the total number of batches (it may increase because batches are not allowed to span across different groups).
        self.n_batch = 0
        self.batch_ids_shuffled = []  # Store all (group_id, batch_id_within_group) pairs.
        for r in range(self.period):
            n_batch_ = int(np.ceil(len(self.sample_ids_grouped[r]) / self.batch_size))
            self.n_batch += n_batch_
            self.batch_ids_shuffled += [(r, i_batch_) for i_batch_ in range(n_batch_)]

    def __iter__(self):
        self.batch_ids_shuffled.sort()
        self.rng.shuffle(self.batch_ids_shuffled)
        return super().__iter__()

    def _get_r_group_i_batch(self, r, i_batch_):
        # Returns the list of sample indices belonging to the i_batch_-th batch of group r.
        sample_ids_ = self.sample_ids_grouped[r]
        list_indices = [
            sample_ids_[i_batch_ * self.batch_size + i] for i in range(self.batch_size)
            if i_batch_ * self.batch_size + i <= len(sample_ids_) - 1]
        return list_indices

    def __next__(self):
        self.i_batch_actual += 1
        if self.i_batch_actual == self.n_batch:
            raise StopIteration()
        r, i_batch_ = self.batch_ids_shuffled[self.i_batch_actual]
        return self._get_r_group_i_batch(r, i_batch_)
