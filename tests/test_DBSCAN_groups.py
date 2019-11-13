import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs

from gridrep import cluster
from helpers.labels import Labels

import pytest

# TODO: MORE TESTS


def create_features(n_samples, centers, n_features):
    X, _ = make_blobs(n_samples=n_samples,
                      centers=centers,
                      n_features=n_features,
                      random_state=0)
    return X


centers_list = [
    [(-2, -2), (0, 0), (4.2, 5)],
    [(-1, -2), (1, 2), (5, 9)]
]


dbscan_parameters = [
    (create_features(n_samples=20000, centers=centers_list[0], n_features=3), 0.1, 3, 2),
    (create_features(n_samples=40000, centers=centers_list[1], n_features=2), 0.2, 10, 1)
]


@pytest.mark.parametrize("features,eps,min_samples,round_decimals", dbscan_parameters)
def test_len_labels(features, eps, min_samples, round_decimals):

    # Cluster features using ClippedDBSCAN
    pipeline_clip = make_pipeline(StandardScaler(),
                                  cluster.ClippedDBSCAN(eps=eps,
                                                        min_samples=min_samples,
                                                        round_decimals=round_decimals))
    labels_clip = Labels('clip', pipeline_clip.fit_predict(features))

    # Cluster features using sklearn DBSCAN
    pipeline_noClip = make_pipeline(StandardScaler(),
                                    FunctionTransformer(np.round,
                                                        validate=False,
                                                        kw_args={"decimals": round_decimals}),
                                    DBSCAN(eps=eps, min_samples=min_samples))
    labels_noClip = Labels('noClip', pipeline_noClip.fit_predict(features))

    assert len(labels_clip.unique.labels) == len(labels_noClip.unique.labels)