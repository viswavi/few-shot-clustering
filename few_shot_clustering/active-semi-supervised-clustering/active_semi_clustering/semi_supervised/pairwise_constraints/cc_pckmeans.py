import math
import numpy as np
from ortools.linear_solver import pywraplp
import time

from active_semi_clustering.semi_supervised.pairwise_constraints.pckmeans import PCKMeans

import sys
sys.path.append("cmvc")
from cmvc.test_performance import cluster_test

class CardinalityConstrainedPCKMeans(PCKMeans):
    def _objective_function(self, X, x_i, centroids, c_i, labels, ml_graph, cl_graph, w, max_pairwise_distance=1.0, print_terms=False):
        distance = 1 / 2 * np.sum((X[x_i] - centroids[c_i]) ** 2)

        ml_violations = 0
        ml_penalty = 0
        for y_i in ml_graph[x_i]:
            if labels[y_i] != -1 and labels[y_i] != c_i:
                # ml_penalty += (max_pairwise_distance - distance)/2
                # assert max_pairwise_distance - distance >= -1e-10
                ml_violations += 1
                ml_penalty += w
#                ml_penalty += distance
        cl_violations = 0
        cl_penalty = 0
        for y_i in cl_graph[x_i]:
            if labels[y_i] == c_i:
                # cl_penalty += distance
                cl_violations += 1
                cl_penalty += w
#                 cl_penalty += distance

        if print_terms:
            metric_dict = {"x_i": x_i, "distance": round(distance, 4), "ml_penalty": round(ml_penalty, 4), "cl_penalty": round(ml_penalty, 4)}
            # print(json.dumps(metric_dict))

        return distance + ml_penalty + cl_penalty, distance, ml_penalty, cl_penalty, ml_violations, cl_violations

    def _assign_clusters(self, X, cluster_centers, ml_graph, cl_graph, w, max_pairwise_distance=1.0):

        # solver = pywraplp.Solver.CreateSolver('SAT')
        solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')
        infinity = solver.infinity()
        objective = solver.Objective()

        avg_points_per_cluster = len(X) / len(cluster_centers)
        min_points_per_cluster = math.floor(avg_points_per_cluster) - 1
        max_points_per_cluster = math.ceil(avg_points_per_cluster) + 1

        labels = np.full(X.shape[0], fill_value=-1)

        _ = '''
        for x_i in range(len(X)):
            cluster_distances = []
            for c_i in range(len(cluster_centers)):
                distance = 1 / 2 * np.sum((X[x_i] - cluster_centers[c_i]) ** 2)
                cluster_distances.append(distance)
            closest_cluster_idx = np.argmin(cluster_distances)
            labels[x_i] = closest_cluster_idx
        '''

        min_cluster_distances = []

        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        point_cluster_weights = []

        variable_creation_start = time.perf_counter()

        variable_matrix = [None for _ in range(X.shape[0])]
        decomposed_assignment_values = [None for _ in range(X.shape[0])]
        for x_i in index:
            objective_composition = [self._objective_function(X, x_i, cluster_centers, c_i, labels, ml_graph, cl_graph, w, max_pairwise_distance=max_pairwise_distance) for c_i in range(self.n_clusters)]
            cluster_distances = [self._objective_function(X, x_i, cluster_centers, c_i, labels, ml_graph, cl_graph, w, max_pairwise_distance=max_pairwise_distance)[0] for c_i in range(self.n_clusters)]
            point_cluster_weights.append(cluster_distances)

            min_cluster_distances.append(min(cluster_distances))
            labels[x_i] = np.argmin(cluster_distances)

            cluster_variables = []
            for c_i in range(self.n_clusters):
                point_cluster_variable = solver.BoolVar(name=f"x_{x_i}->c_{c_i}")
                cluster_variables.append(point_cluster_variable)
                objective.SetCoefficient(point_cluster_variable, cluster_distances[c_i])
            variable_matrix[x_i] = cluster_variables
            decomposed_assignment_values[x_i] = objective_composition
        decomposed_assignment_values = np.array(decomposed_assignment_values)

        _ = '''
distance_values = decomposed_assignment_values[:, :, 1]
ml_violations_values = decomposed_assignment_values[:, :, 2]
cl_violations_values = decomposed_assignment_values[:, :, 3]
ml_violations_counts = np.sum(decomposed_assignment_values[:, :, 4], axis=1)
cl_violations_counts = np.sum(decomposed_assignment_values[:, :, 5], axis=1)
ml_violations_count_means = np.mean(decomposed_assignment_values[:, :, 4], axis=1)
cl_violations_count_means = np.mean(decomposed_assignment_values[:, :, 5], axis=1)

cluster_assignments = np.argmin(distance_values, axis=1)
kpp_cluster_assignments = np.argmin(euclidean_distances(X, cluster_centers), axis=1)
ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, macro_f1, micro_f1, pairwise_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons  = cluster_test(self.side_information.p, self.side_information.side_info, kpp_cluster_assignments, self.side_information.true_ent2clust, self.side_information.true_clust2ent)
ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, macro_f1, micro_f1, pairwise_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons  = cluster_test(self.side_information.p, self.side_information.side_info, cluster_assignments, self.side_information.true_ent2clust, self.side_information.true_clust2ent)
        '''

        variable_creation_end = time.perf_counter()

        constraint_creation_start = time.perf_counter()
        point_assignment_constraints = []
        for x_i in range(len(X)):
            single_assignment_constraint = solver.RowConstraint(1, 1, str(f"x_{x_i}"))
            point_assignment_constraints.append(single_assignment_constraint)

        cluster_size_constraints = []
        for c_i in range(self.n_clusters):
            cardinality_constraint = solver.RowConstraint(2, 20, str(f"c_{c_i}"))
            #  cardinality_constraint = solver.RowConstraint(2, infinity, str(f"c_{c_i}"))
            for x_i in range(len(X)):
                assignment = variable_matrix[x_i][c_i]
                cardinality_constraint.SetCoefficient(assignment, 1)
                point_assignment_constraints[x_i].SetCoefficient(assignment, 1)
            cluster_size_constraints.append(cardinality_constraint)
        constraint_creation_end = time.perf_counter()

        assert solver.NumConstraints() == self.n_clusters + len(X)
        objective.SetMinimization()
        constraint_solving_start = time.perf_counter()
        status = solver.Solve()
        constraint_solving_end = time.perf_counter()

        if status == pywraplp.Solver.OPTIMAL:
            for var_idx, assignment_list in enumerate(variable_matrix):
                assignment = None
                for clust_idx, clust_variable in enumerate(assignment_list):
                    if clust_variable.solution_value() == 1.0:
                        assignment = clust_idx
                        break
                assert assignment is not None
                labels[var_idx] = assignment
        else:
            breakpoint()
        return labels

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
