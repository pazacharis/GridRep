import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field


class Labels:
    def __init__(self, name, labels: np.array):
        self.name = name
        self.labels = labels
        self.unique = Unique(*np.unique(self.labels, return_counts=True))

    def sort_counts(self):
        return self.unique.counts[self.unique.indices_sort_counts]

    def sort_unique_values(self):
        return self.unique.labels[self.unique.indices_sort_counts]


@dataclass
class Unique:
    labels: np.array
    counts: np.array
    indices_sort_counts: np.array = field(init=False)

    def __post_init__(self):
        self.indices_sort_counts = np.argsort(self.counts)[::-1]


def where_not_all_match(labels_1, labels_2):
    to_return = []
    for unique_label_1 in labels_1.unique.labels:
        idx_1 = np.where(labels_1.labels == unique_label_1)[0]

        corresponding_labels_2 = labels_2.labels[idx_1]
        if not all(corresponding_labels_2 == corresponding_labels_2[0]):
            to_return.append([unique_label_1, idx_1, corresponding_labels_2])
    return to_return


def scatter_from_mismatch(labels1, labels2,
                          labels1_labels2,
                          X, excluded=None):
    if excluded is None:
        excluded = []

    opt = {'noClip': {'marker': 'D',
                      'facecolors': 'none',
                      'size': 70},
           'clip': {'marker': '.',
                    'facecolors': np.random.rand(3, ),
                    'size': 100}}

    plotted = []
    for label in labels1_labels2:
        label1 = label[0]
        idx1 = label[1]

        local_label = _create_label_id(labels1.name, label1)
        if local_label not in excluded:
            plotted.append(local_label)
            plt.scatter(X[idx1, 0], X[idx1, 1],
                        label=local_label,
                        marker=opt[labels1.name]['marker'],
                        alpha=0.5,
                        facecolors=opt[labels1.name]['facecolors'],
                        edgecolors=np.random.rand(3, ),
                        s=100)

        corresponding_labels = np.unique(label[2])
        for corr_label in corresponding_labels:
            idx2 = np.where(labels2.labels == corr_label)[0]
            local_label = _create_label_id(labels2.name, corr_label)
            if local_label not in excluded:
                plotted.append(local_label)
                plt.scatter(X[idx2, 0], X[idx2, 1],
                            label=local_label,
                            marker=opt[labels2.name]['marker'],
                            alpha=0.5,
                            facecolors=opt[labels2.name]['facecolors'],
                            edgecolors=np.random.rand(3, ),
                            s=100)
    return plotted


def _create_label_id(name, label):
    return name + '_' + str(label)