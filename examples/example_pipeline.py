from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets.samples_generator import make_blobs

import sys
sys.path.append("..")

from gridrep.cluster import ClippedDBSCAN


def blobs(centers=None):
    if centers is None:
        centers = [(-2, -2), (0, 0), (4.2, 5)]
    X, y = make_blobs(n_samples=20000, centers=centers, n_features=2, random_state=0)
    return X


def full_pipeline(X=None, radius=0.1, min_samples=3, round_decimals=1, scaler=StandardScaler()):
    if X is None:
        X = blobs()
    pipeline_clip = make_pipeline(scaler, ClippedDBSCAN(eps=radius,
                                                        min_samples=min_samples,
                                                        round_decimals=round_decimals))
    labels_clip = pipeline_clip.fit_predict(X)
    return labels_clip


if __name__ == "__main__":
    
    _ = full_pipeline()
    print('Pipeline with ClippedDBSCAN completed.')
