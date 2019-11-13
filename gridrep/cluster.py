from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, ClusterMixin
from typing import Optional
from gridrep.preprocess import FeaturesTransformer


class ClippedDBSCAN(BaseEstimator, ClusterMixin):
    """Perform DBSCAN clustering with 'just-enough' representative
    feature transformation and label remapping.

    Currently it wraps scikit-learn's DBSCAN.
    """
    def __init__(self,
                 eps: float,
                 min_samples: int,
                 round_decimals: Optional[int] = None):
        self.round_decimals = round_decimals
        self.eps = eps
        self.min_samples = min_samples

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
                       algorithm="ball_tree").fit(X)
        return model
