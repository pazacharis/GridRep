import os
import numpy as np
import time

from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs

import gridrep


def test_counts():
    centers = [(-2, -2), (0, 0), (4.2, 5)]
    X, y = make_blobs(n_samples=20000, centers=centers, n_features=2, random_state=0)

    # Reduce cardinality by rounding
    decimals = 1

    # DBSCAN parameters
    radius = 0.1
    min_samples = 3

    model_clipped = gridrep.cluster.ClippedDBSCAN(eps=radius, min_samples=min_samples)
    labels_clipped = model_clipped.fit_predict(X)

    X_rounded = np.round(X, decimals=decimals)
    model_noclip = DBSCAN(eps=radius, min_samples=min_samples).fit(X_rounded)
    labels_noclip = model_noclip.labels_

    counts_all = []
    for _labels_ in [labels_clipped, labels_noclip]:
        _, counts = np.unique(_labels_, return_counts=True)
        counts_sorted = np.sort(counts)[::-1]
        print(counts_sorted[:10])

        counts_all.append(counts_sorted)

    assert all(counts_all[0]==counts_all[1])