#!/usr/bin/env python

# external
import joblib
from tqdm.auto import tqdm

# from https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib/61027781#61027781

class ProgressParallel(joblib.Parallel):
    """Joblib Parallel with monitoring
    """
    def __call__(self, *args, **kwargs):
        with tqdm(position = 0, desc = 'Parallel') as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
