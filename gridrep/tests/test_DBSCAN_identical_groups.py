import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs

from gridrep import cluster


def test_counts():

    centers = [(-2, -2), (0, 0), (4.2, 5)]
    X, y = make_blobs(n_samples=20000, centers=centers, n_features=2, random_state=0)

    # Reduce cardinality by rounding
    round_decimals = 1

    # DBSCAN parameters
    radius = 0.1
    min_samples = 3

    pipeline_clip = make_pipeline(StandardScaler(),
                                  cluster.ClippedDBSCAN(eps=radius,
                                                        min_samples=min_samples,
                                                        round_decimals=round_decimals))
    labels_clip = pipeline_clip.fit_predict(X)

    pipeline_noClip = make_pipeline(StandardScaler(),
                                    FunctionTransformer(np.round,
                                                        validate=False,
                                                        kw_args={"decimals": round_decimals}),
                                    DBSCAN(eps=radius, min_samples=min_samples))
    labels_noClip = pipeline_noClip.fit_predict(X)

    counts_all = []
    for _labels_ in [labels_clip, labels_noClip]:
        _, counts = np.unique(_labels_, return_counts=True)
        counts_sorted = np.sort(counts)[::-1]
        print(counts_sorted[:10])

        counts_all.append(counts_sorted)

    assert all(counts_all[0]==counts_all[1])