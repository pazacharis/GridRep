import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class _Representatives:
    rows_representatives: np.ndarray
    unique_inverse: np.ndarray
    len_added_rows: int


class FeaturesTransformer:
    """Transform array and generate 'just-enough' representatives for DBSCAN clustering.
    Use remap_labels method for remapping labels to original array.

    Parameters
    ----------
    features : array (n_samples, n_features)
    min_samples : int
        DBSCAN min_samples value.
    round_decimals : int, or None
        If a value is provided, the features will be rounded to the given number of decimals.
        Use to reduce false precision (i.e. meaningless decimals).

    Attributes
    ----------
    representatives : _Representatives
        Contains generated representative array, indices for unique inversing
        and the length difference between representatives and unique rows (required for remapping).
    """
    def __init__(self, features: np.ndarray, min_samples: int, round_decimals: Optional[int] = None):
        self.round_decimals = round_decimals
        self.min_samples = min_samples

        self.features = self._features_prep(features)
        self.representatives = self._representatives()

    def remap_labels(self, labels_of_stacked_rows: np.ndarray) -> np.ndarray:
        """
        Remap DBSCAN labels of representatives back to original feature set.

        Parameters
        -----------
        labels_of_stacked_rows : array (n_samples_representatives,)

        Returns
        --------
        labels_remapped : array (n_samples, )
        """
        labels_of_unique_rows = labels_of_stacked_rows[:-self.representatives.len_added_rows]

        labels_remapped = labels_of_unique_rows[self.representatives.unique_inverse]
        return labels_remapped

    def _representatives(self) -> _Representatives:
        """Generate a _Representatives object
        """
        unique_rows, unique_inverse, unique_counts = np.unique(self.features,
                                                               axis=0,
                                                               return_inverse=True,
                                                               return_counts=True)

        unique_counts_clipped_upto_min_samples = np.clip(unique_counts, 1, self.min_samples + 1) - 1
        rows_to_stack = unique_rows.repeat(unique_counts_clipped_upto_min_samples, axis=0)
        unique_rows_appended_with_repeats = np.append(unique_rows, rows_to_stack, axis=0)

        return _Representatives(unique_rows_appended_with_repeats, unique_inverse, len(rows_to_stack))

    def _round_(self, decimals: int) -> np.ndarray:
        rounded = np.round(self.features, decimals=decimals)
        return rounded

    def _features_prep(self, features: np.ndarray) -> np.ndarray:
        if isinstance(self.round_decimals, int):
            return np.round(features, self.round_decimals)
        return features



