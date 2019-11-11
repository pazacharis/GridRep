import numpy as np
from typing import NamedTuple


class FeaturesTransformer():
    def __init__(self, features, min_samples, round_decimals=None):
        self.round_decimals = round_decimals
        self.min_samples = min_samples

        self.features = self._features_prep(features)
        self.representatives = self._representatives()

    def remap_labels(self, labels):
        labels_clipped = labels[:-self.representatives.len_added_rows]
        labels_remapped = labels_clipped[self.representatives.unique_inverse]
        return labels_remapped

    def _representatives(self):
        unique_rows, unique_inverse, unique_counts = np.unique(self.features,
                                                               axis=0,
                                                               return_inverse=True,
                                                               return_counts=True)

        counts_clipped = np.clip(unique_counts, 1, self.min_samples + 1) - 1
        rows_to_stack = unique_rows.repeat(counts_clipped, axis=0)
        unique_X_stacked = np.append(unique_rows, rows_to_stack, axis=0)

        return _Representatives(unique_X_stacked, unique_inverse, len(rows_to_stack))

    def _round_(self, decimals):
        rounded = np.round(self.features, decimals=decimals)
        return rounded

    def _features_prep(self, features):
        if isinstance(self.round_decimals, int):
            return np.round(features, self.round_decimals)
        return features


class _Representatives(NamedTuple):
    rows_representatives: np.array
    unique_inverse: np.array
    len_added_rows: int
