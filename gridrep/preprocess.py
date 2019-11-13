import numpy as np
# from typing import NamedTuple
from dataclasses import dataclass


@dataclass
class _Representatives:
    rows_representatives: np.array
    unique_inverse: np.array
    len_added_rows: int


class FeaturesTransformer:
    def __init__(self, features, min_samples, round_decimals=None):
        self.round_decimals = round_decimals
        self.min_samples = min_samples

        self.features = self._features_prep(features)
        self.representatives = self._representatives()

    def remap_labels(self, labels_of_stacked_rows: np.array) -> np.array:
        labels_of_unique_rows = labels_of_stacked_rows[:-self.representatives.len_added_rows]

        labels_remapped = labels_of_unique_rows[self.representatives.unique_inverse]
        return labels_remapped

    def _representatives(self) -> _Representatives:
        unique_rows, unique_inverse, unique_counts = np.unique(self.features,
                                                               axis=0,
                                                               return_inverse=True,
                                                               return_counts=True)

        unique_counts_clipped_upto_min_samples = np.clip(unique_counts, 1, self.min_samples + 1) - 1
        rows_to_stack = unique_rows.repeat(unique_counts_clipped_upto_min_samples, axis=0)
        unique_rows_appended_with_repeats = np.append(unique_rows, rows_to_stack, axis=0)

        return _Representatives(unique_rows_appended_with_repeats, unique_inverse, len(rows_to_stack))

    def _round_(self, decimals: int) -> np.array:
        rounded = np.round(self.features, decimals=decimals)
        return rounded

    def _features_prep(self, features: np.int) -> np.int:
        if isinstance(self.round_decimals, int):
            return np.round(features, self.round_decimals)
        return features



