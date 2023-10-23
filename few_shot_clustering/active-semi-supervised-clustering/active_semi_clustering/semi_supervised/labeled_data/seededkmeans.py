from collections import defaultdict
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
import sys

from .kmeans import KMeans

from cmvc.Multi_view_CH_kmeans import init_seeded_kmeans_plusplus


class SeededKMeans(KMeans):
    def _init_cluster_centers(self, X, y=None, random_seed=0):
        assert y is not None and not np.all(y == -1)
        random_state = check_random_state(random_seed)
        x_squared_norms = row_norms(X, squared=True)

        seed_labels = set([y_value for y_value in y if y_value != -1])
        seeded_feature_clusters = [[X[i] for i, y_value in enumerate(y) if y_value == label] for label in seed_labels]
        seed_set = np.vstack([np.mean(np.vstack(cluster), axis=0) for cluster in seeded_feature_clusters])

        if self.init == "k-means++":
            seeds = super()._init_cluster_centers(X, seed_set=seed_set)
        else:
            remaining_seeds_available = list(range(len(X)))
            remaining_seeds_chosen = np.random.choice(remaining_seeds_available, size=self.n_clusters - len(seed_set), replace=False)
            seeds = np.vstack([seed_set, X[remaining_seeds_chosen]])
        return seeds
