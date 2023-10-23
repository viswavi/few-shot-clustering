import copy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from .example_oracle import MaximumQueriesExceeded
from .explore_consolidate import ExploreConsolidate


class MinMax(ExploreConsolidate):
    def _consolidate(self, neighborhoods, X, oracle):
        n = X.shape[0]

        skeleton = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                skeleton.add(i)

        remaining = set()
        for i in range(n):
            if i not in skeleton:
                remaining.add(i)

        distances = euclidean_distances(X, X)
        kernel_width = np.percentile(distances, 20)

        pairwise_distances = euclidean_distances(X, X, squared=True)
        kernel_similarities = np.exp(-pairwise_distances / (2 * (kernel_width ** 2)))

        while True:
            try:
                max_similarities = np.full(n, fill_value=float('+inf'))
                
                for x_i in remaining:
                    max_similarities[x_i] = np.max(kernel_similarities[x_i, list(skeleton)])

                q_i = max_similarities.argmin()

                sorted_neighborhoods = reversed(sorted(neighborhoods, key=lambda neighborhood: np.max(kernel_similarities[q_i, list(neighborhood)])))

                for neighborhood in sorted_neighborhoods:
                    self.counts += 1
                    print(f"Consolidate Count: {self.counts}")

                    if oracle.query(q_i, neighborhood[0]):
                        neighborhood.append(q_i)
                        break

                skeleton.add(q_i)
                if len(remaining) == 0:
                    return neighborhoods
                else:
                    remaining.remove(q_i)

            except MaximumQueriesExceeded:
                break

        return neighborhoods


class SimilarityFinder(ExploreConsolidate):
    def fit(self, X, oracle=None):
        if oracle.max_queries_cnt <= 0:
            return [], []

        neighborhoods = self._explore(X, self.n_clusters, oracle)
        self.pairwise_constraints_ = self._consolidate(neighborhoods, X, oracle)

        return self

    def _consolidate(self, neighborhoods, X, oracle):
        ml = []

        for neighborhood in neighborhoods:
            for i in neighborhood:
                for j in neighborhood:
                    if i != j:
                        ml.append((i, j))

        cl = []
        for neighborhood in neighborhoods:
            for other_neighborhood in neighborhoods:
                if neighborhood != other_neighborhood:
                    for i in neighborhood:
                        for j in other_neighborhood:
                            cl.append((i, j))

        n = X.shape[0]

        neighborhoods_union = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        remaining = set()
        for i in range(n):
            if i not in neighborhoods_union:
                remaining.add(i)

        pairwise_distances = euclidean_distances(X, X, squared=True)

        neighborhoods_observed = copy.deepcopy(neighborhoods)
        neighborhoods_observed = [set(n) for n in neighborhoods_observed]

        while True:

            try:
                neighborhood_idx = np.random.choice(list(range(len(neighborhoods))))
                member = np.random.choice(neighborhoods[neighborhood_idx])
                distances = pairwise_distances[member]

                local_remaining = remaining - neighborhoods_observed[neighborhood_idx]

                remaining_list = list(local_remaining)
                remaining_distances = distances[remaining_list]
                if len(remaining_distances) == 0:
                    breakpoint()
                closest_point = remaining_list[np.argmin(remaining_distances)]

                oracle_response = oracle.query(member, closest_point)
                if oracle_response is True:
                    ml.append((member, closest_point))
                    neighborhoods[neighborhood_idx].append(closest_point)
                elif oracle_response is False:
                    cl.append((member, closest_point))
                neighborhoods_observed[neighborhood_idx].add(closest_point)
                self.counts += 1
                print(f"Consolidate Count: {self.counts}")

            except MaximumQueriesExceeded:
                break

        return ml, cl


def similarity(x, y, kernel_width):
    return np.exp(-((x - y) ** 2).sum() / (2 * (kernel_width ** 2)))
