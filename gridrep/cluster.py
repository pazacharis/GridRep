from sklearn.cluster import DBSCAN
from sklearn.base import ClusterMixin

from . import preprocess


class ClippedDBSCAN(ClusterMixin):
    def __init__(self, eps, min_samples, round_decimals=1, algorithm="ball_tree"):
        self.round_decimals = round_decimals
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm

    def fit_predict(self, X, y=None):
        X_rep, unique_inverse, len_additional = self._transform(X)

        model = self._fit(X_rep)

        labels_remapped = preprocess.remap_labels(model.labels_, len_additional, unique_inverse)

        return labels_remapped

    def _transform(self, X):
        X_rounded = preprocess.round_(X, self.round_decimals)

        rep = preprocess.representatives(X_rounded, self.min_samples)

        X_rep = rep['X_representatives']
        unique_inverse = rep['unique_inverse']
        len_additional = rep['len_discard_rows']

        return X_rep, unique_inverse, len_additional

    def _fit(self, X, y=None):
        model = DBSCAN(eps=self.eps,
                       min_samples=self.min_samples,
                       algorithm=self.algorithm).fit(X)
        return model






