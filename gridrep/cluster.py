from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, ClusterMixin

from gridrep.preprocess import FeaturesTransformer


class ClippedDBSCAN(ClusterMixin):
    def __init__(self, eps, min_samples, round_decimals=None, algorithm="ball_tree"):
        self.round_decimals = round_decimals
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm

        self.labels_ = None

    def fit(self, X, y=None):
        participating = FeaturesTransformer(X,
                                            self.min_samples,
                                            self.round_decimals)

        model = self._fit(participating.representatives.rows_representatives)

        self.labels_ = participating.remap_labels(model.labels_)

        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def _fit(self, X, y=None):
        model = DBSCAN(eps=self.eps,
                       min_samples=self.min_samples,
                       algorithm=self.algorithm).fit(X)
        return model


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.datasets.samples_generator import make_blobs

    radius = 0.1
    min_samples = 3
    round_decimals = 1

    centers = [(-2, -2), (0, 0), (4.2, 5)]
    X, y = make_blobs(n_samples=20000, centers=centers, n_features=2, random_state=0)

    pipeline_clip = make_pipeline(StandardScaler(), ClippedDBSCAN(eps=radius,
                                                                  min_samples=min_samples,
                                                                  round_decimals=round_decimals))
    labels_clip = pipeline_clip.fit_predict(X)
    print('Pipeline completed.')


