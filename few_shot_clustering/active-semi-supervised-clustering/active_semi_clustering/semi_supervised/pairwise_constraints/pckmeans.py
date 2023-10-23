import json
import math
import numpy as np
from ortools.linear_solver import pywraplp
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import time

from active_semi_clustering.exceptions import EmptyClustersException
from .constraints import preprocess_constraints, preprocess_constraints_no_transitive_closure
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans

import sys
sys.path.append("cmvc")
from cmvc.test_performance import cluster_test

class PCKMeans(KMeans):
    def __init__(self, n_clusters=3, max_iter=100, w=0.25, init="random", normalize_vectors=False, split_normalization=False, side_information=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w
        self.init = init
        self.normalize_vectors = normalize_vectors
        self.split_normalization = split_normalization
        self.side_information = side_information

    def fit(self, X, y=None, ml=[], cl=[]):
        # Preprocess constraints
        # _, _, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])
        ml_graph, cl_graph = preprocess_constraints_no_transitive_closure(ml, cl, X.shape[0])

        print(f"ML constraints:\n{ml}\n")
        print(f"CL constraints:\n{cl}\n")

        # print(f"Num neighborhoods: {sorted([len(n) for n in neighborhoods])}\n\n\n")

        # Initialize centroids
        start = time.perf_counter()
        cluster_centers = self._init_cluster_centers(X)
        # cluster_centers = self._initialize_cluster_centers(X, neighborhoods)
        elapsed = time.perf_counter() - start
        print(f"Initializing neighborhoods took {round(elapsed, 4)} seconds")


        # Repeat until convergence
        for iteration in range(self.max_iter):
            print(f"\n\n\n\niteration: {iteration}")


            if self.normalize_vectors:
                if self.split_normalization:
                    kg_centers = normalize(cluster_centers[:, :300], axis=1, norm="l2")
                    bert_centers = normalize(cluster_centers[:, 300:], axis=1, norm="l2")
                    cluster_centers = np.hstack([kg_centers, bert_centers])
                else:
                    cluster_centers = normalize(cluster_centers, axis=1, norm="l2")

            start = time.perf_counter()
            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, ml_graph, cl_graph, self.w)

            # Estimate means
            prev_cluster_centers = cluster_centers
            cluster_centers = self._get_cluster_centers(X, labels)

            # Check for convergence
            difference = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(difference, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)
            elapsed = time.perf_counter() - start
            print(f"elapsed time: {round(elapsed, 3)}")

            if self.side_information is not None and not isinstance(self.side_information, list) and iteration % 10 == 0:
                ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, macro_f1, micro_f1, pairwise_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons  = cluster_test(self.side_information.p, self.side_information.side_info, labels, self.side_information.true_ent2clust, self.side_information.true_clust2ent)
                metric_dict = {"macro_f1": macro_f1, "micro_f1": micro_f1, "pairwise_f1": pairwise_f1, "ave_f1": ave_f1}
                print(f"metric_dict at iteration {iteration}:\t{metric_dict}")

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])

        print("Initializing cluster centers")
        if len(neighborhoods) > self.n_clusters:
            # Select K largest neighborhoods' centroids
            cluster_centers = neighborhood_centers[np.argsort(neighborhood_sizes)[-self.n_clusters:]]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            # FIXME look for a point that is connected by cannot-links to every neighborhood set

            if len(neighborhoods) < self.n_clusters:
                if self.init == "k-means++":
                    print("Running K-Means++")
                    if len(list(cluster_centers)) > 0:
                        cluster_centers = super()._init_cluster_centers(X, seed_set=list(cluster_centers))
                    else:
                        cluster_centers = super()._init_cluster_centers(X)
                else:
                    remaining_cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters - len(neighborhoods), replace=False), :]
                    cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])
        return cluster_centers

    def _objective_function(self, x_i, point_cluster_distances, c_i, labels, ml_graph, cl_graph, w, print_terms=False):
        distance = 1 / 2 * point_cluster_distances[x_i, c_i]

        ml_penalty = 1
        for y_i in ml_graph[x_i]:
            if labels[y_i] != -1 and labels[y_i] != c_i:
                # ml_penalty += (max_pairwise_distance - distance)/2
                # assert max_pairwise_distance - distance >= -1e-10
                ml_penalty += w
#                ml_penalty += distance

        cl_penalty = 1
        for y_i in cl_graph[x_i]:
            if labels[y_i] == c_i:
                # cl_penalty += distance
                cl_penalty += w
#                 cl_penalty += distance

        if print_terms:
            metric_dict = {"x_i": x_i, "distance": round(distance, 4), "ml_penalty": round(ml_penalty, 4), "cl_penalty": round(ml_penalty, 4)}
            # print(json.dumps(metric_dict))

        return distance + ml_penalty + cl_penalty, ml_penalty, cl_penalty

    def _assign_clusters(self, X, cluster_centers, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1)
        min_cluster_distances = []

        index = list(range(X.shape[0]))
        point_cluster_distances = euclidean_distances(X, cluster_centers, squared=True)
        np.random.shuffle(index)

        for x_i in index:
            cluster_distances = [self._objective_function(x_i, point_cluster_distances, c_i, labels, ml_graph, cl_graph, w)[0] for c_i in range(self.n_clusters)]
            min_cluster_distances.append(min(cluster_distances))
            labels[x_i] = np.argmin(cluster_distances)

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        continue_counter = 0
        while len(empty_clusters) > 0:
            original_labels = labels.copy()
            print(f"Empty clusters: {empty_clusters}")
            points_by_min_cluster_distance = np.argsort(-np.array(min_cluster_distances))
            i = 0
            for cluster_idx in list(empty_clusters):
                while n_samples_in_cluster[labels[points_by_min_cluster_distance[i]]] == 1:
                    i += 1
                labels[points_by_min_cluster_distance[i]] = cluster_idx
                i += 1

            n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
            empty_clusters = np.where(n_samples_in_cluster == 0)[0]
            if len(empty_clusters) > 0:
                continue_counter += 1
            if continue_counter > 10:
                breakpoint()
        return labels

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
