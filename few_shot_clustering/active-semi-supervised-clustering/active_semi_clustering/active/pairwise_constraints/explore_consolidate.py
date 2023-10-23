import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time

from .helpers import get_constraints_from_neighborhoods
from .example_oracle import MaximumQueriesExceeded


class ExploreConsolidate:
    def __init__(self, n_clusters=3, **kwargs):
        self.n_clusters = n_clusters
        self.counts = 0

    def fit(self, X, oracle=None):
        if oracle.max_queries_cnt <= 0:
            return [], []

        neighborhoods = self._explore(X, self.n_clusters, oracle)
        neighborhoods = self._consolidate(neighborhoods, X, oracle)

        self.pairwise_constraints_ = get_constraints_from_neighborhoods(neighborhoods)

        return self

    def _explore(self, X, k, oracle, max_explore_ratio=0.5):
        neighborhoods = []
        traversed = []
        n = X.shape[0]

        x = np.random.choice(n)
        neighborhoods.append([x])
        traversed.append(x)

        total_start = time.perf_counter()

        try:
            while len(neighborhoods) < k:
                farthest = None

                candidate_indices = [i for i in range(n) if i not in traversed]  
                distances = batch_dist(candidate_indices, traversed, X)
                farthest = candidate_indices[np.argmax(distances)]

                new_neighborhood = True
                for neighborhood in neighborhoods:
                    self.counts += 1
                    print(f"Count: {self.counts}")
                    if oracle.query(farthest, neighborhood[0]):
                        neighborhood.append(farthest)
                        new_neighborhood = False
                        break

                if new_neighborhood:
                    neighborhoods.append([farthest])

                traversed.append(farthest)

                if oracle.queries_cnt / oracle.max_queries_cnt > max_explore_ratio:
                    break


        except MaximumQueriesExceeded:
            pass


        return neighborhoods

    def _consolidate(self, neighborhoods, X, oracle):
        n = X.shape[0]

        neighborhoods_union = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        remaining = set()
        for i in range(n):
            if i not in neighborhoods_union:
                remaining.add(i)

        while True:

            try:
                i = np.random.choice(list(remaining))

                sorted_neighborhoods = sorted(neighborhoods, key=lambda neighborhood: dist(i, neighborhood, X))

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(i, neighborhood[0]):
                        neighborhood.append(i)
                        self.counts += 1
                        print(f"Count: {self.counts}")
                        break

                neighborhoods_union.add(i)
                remaining.remove(i)

            except MaximumQueriesExceeded:
                break

        return neighborhoods


def dist(i, S, points):
    distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in S])
    return distances.min()


def batch_dist(Is, S, points):
    pairwise_distances = euclidean_distances(points[Is], points[S])
    min_distances = np.min(pairwise_distances, axis=1)
    return min_distances