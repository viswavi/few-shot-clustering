import numpy as np

from .example_oracle import MaximumQueriesExceeded
from .explore_consolidate import ExploreConsolidate
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans

class MinMaxFinetune(ExploreConsolidate):
    def set_initial_clusterer(self, initial_clusterer):
        self._initial_clusterer = initial_clusterer

    def _explore(self, X, k, oracle):
        cluster_predict_list = self._initial_clusterer.labels_
        cluster_assignments = {}
        for i, cluster_id in enumerate(cluster_predict_list):
            cluster_elements = cluster_assignments.get(cluster_id, [])
            cluster_elements.append(i)
            cluster_assignments[cluster_id] = cluster_elements
        neighborhoods = []
        for cluster_id, cluster_elements in cluster_assignments.items():
            cluster_element_vectors = [X[el] for el in cluster_elements]
            centroid = np.mean(np.vstack(cluster_element_vectors), axis=0)
            element_distances = [np.linalg.norm(X[el] - centroid) for el in cluster_elements]
            closest_element = cluster_elements[np.argmin(element_distances)]
            neighborhoods.append([closest_element])
        return neighborhoods

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

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.sqrt(((X[i] - X[j]) ** 2).sum())

        kernel_width = np.percentile(distances, 20)

        counter = 0
        while True:
            try:
                max_similarities = np.full(n, fill_value=float('+inf'))
                for x_i in remaining:
                    max_similarities[x_i] = np.max([similarity(X[x_i], X[x_j], kernel_width) for x_j in skeleton])

                q_i = max_similarities.argmin()

                sorted_neighborhoods = reversed(sorted(neighborhoods, key=lambda neighborhood: np.max([similarity(X[q_i], X[n_i], kernel_width) for n_i in neighborhood])))

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(q_i, neighborhood[0]):
                        neighborhood.append(q_i)
                        break

                skeleton.add(q_i)
                remaining.remove(q_i)
                print(f"Counter: {counter}")
                counter += 1

                if len(remaining) == 0:
                    breakpoint()

            except MaximumQueriesExceeded:
                break

        return neighborhoods


def similarity(x, y, kernel_width):
    return np.exp(-((x - y) ** 2).sum() / (2 * (kernel_width ** 2)))
