import math
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .example_oracle import MaximumQueriesExceeded
from .explore_consolidate import ExploreConsolidate


class DistanceBasedSelector:
    def __init__(self, n_clusters=3, rerank_ratio = 8, batch_size=1250, **kwargs):
        self.n_clusters = n_clusters
        self.rerank_ratio = rerank_ratio
        self.batch_size = batch_size

    def choose_closest_point(self, X, sample_idx, other_indices):
        distances = np.linalg.norm(X[sample_idx] - X[other_indices], axis=1)
        return other_indices[np.argmin(distances)]

    def choose_furthest_point(self, X, sample_idx, other_indices):
        distances = np.linalg.norm(X[sample_idx] - X[other_indices], axis=1)
        return other_indices[np.argmax(distances)]

    def fit(self, X, oracle=None):
        print(f"Selecting constraints: oracle.max_queries_cnt: {oracle.max_queries_cnt}")
        if oracle.max_queries_cnt <= 0:
            return [], []

        X_normalized = np.hstack([normalize(X[:, :300], axis=1, norm="l2"), normalize(X[:, 300:], axis=1, norm="l2")])

        ml = []
        cl = []

        # choose point pairs that are close together

        print("0")
        x_squared_norms = row_norms(X_normalized, squared=True)
        print("1")
        distance_matrix = euclidean_distances(X_normalized, X_normalized, Y_norm_squared=x_squared_norms, squared=True)
        print("2")
        distance_matrix_flattened = np.ravel(distance_matrix)
        print("3")
        flattened_matrix_sort_indices_unfiltered = np.argsort(distance_matrix_flattened)
        print("4")
        matrix_sort_indices = []
        for ind in flattened_matrix_sort_indices_unfiltered:
            (x,y) = (ind // len(X_normalized), ind % len(X_normalized))
            try:
                if x < y and oracle.documents[x] != oracle.documents[y]:
                    matrix_sort_indices.append((x,y))
            except:
                breakpoint()
        print("5")

        print(f"Pseudo-labeling constraints")
        reranked_indices = matrix_sort_indices[:oracle.max_queries_cnt]
        for x, y in tqdm(reranked_indices):
            pair_label = oracle.query(x, y)
            if pair_label == True:
                ml.append([x, y])
            elif pair_label == False:
                cl.append([x, y])

        self.pairwise_constraints_ = (ml, cl)

        return self
