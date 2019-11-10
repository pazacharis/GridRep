import numpy as np


def round_(X, decimals):
    X_rounded = np.round(X, decimals=decimals)
    return X_rounded


def representatives(X, min_samples):
    X_unique_rows, unique_inverse, unique_counts = np.unique(X,
                                                             axis=0,
                                                             return_inverse=True,
                                                             return_counts=True)

    counts_clipped = np.clip(unique_counts, 1, min_samples + 1) - 1
    rows_to_stack = X_unique_rows.repeat(counts_clipped, axis=0)
    unique_X_stacked = np.append(X_unique_rows, rows_to_stack, axis=0)

    to_return = {'X_representatives': unique_X_stacked,
                 'unique_inverse': unique_inverse,
                 'len_discard_rows': len(rows_to_stack)}

    return to_return


def remap_labels(labels, len_additional, unique_inverse):
    labels_clipped = labels[:-len_additional]
    labels_remapped = labels_clipped[unique_inverse]
    return labels_remapped
