from __future__ import division, generators, print_function, unicode_literals, with_statement

import os
import numpy as np

from data.persistence import DataPersistence
from data.feature_loader import FeatureLoader

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class FeatureSequenceLoader(FeatureLoader):

    def __init__(self, n_steps=10, **kwargs):
        self.n_steps = n_steps
        super().__init__(**kwargs)

        # Generate indices list
        idx_ranges = [(i*1000+self.n_steps, (i+1)*1000) for i in range(0, self.frame_count // 1000)]

        self.frame_seq_indices = []
        for idx_min, idx_max in idx_ranges:
            for idx in range(idx_min, idx_max + 1):
                self.frame_seq_indices.append(
                    np.arange(idx - self.n_steps, idx)
                )
        self.frame_seq_indices = np.asarray(self.frame_seq_indices)


    def __iter__(self):
        print('FeatureSequenceLoader __iter__ called')

        for indices in self.frame_seq_indices:
            yield self.inputs_memmap[indices], self.targets_memmap[indices[-1]]
