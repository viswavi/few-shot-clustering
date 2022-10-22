# python active_clustering.py --dataset iris --num_clusters 3 --num_seeds 10
# python active_clustering.py --dataset 20_newsgroups --data-path data/20_newsgroups/

from sklearn import datasets, metrics

import argparse
from collections import defaultdict
import json
import numpy as np
import os
import random
import sys

from datasets import load_dataset
from utils import set_seed, summarize_results

sys.path.extend(["..", "."])

if os.getenv("REPO_DIR") is not None:
    sys.path.append(os.path.join(os.getenv("REPO_DIR"), "clustering", "active-semi-supervised-clustering"))
else:
    sys.path.append("active-semi-supervised-clustering")
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans
from active_semi_clustering.semi_supervised.labeled_data.seededkmeans import SeededKMeans
from active_semi_clustering.semi_supervised.labeled_data.constrainedkmeans import ConstrainedKMeans
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, MinMax


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["iris", "20_newsgroups"], default="iris", help="Clustering dataset to experiment with")
parser.add_argument('--data-path', type=str, default=None, help="Path to clustering data, if necessary")
parser.add_argument('--num_clusters', type=int, default=3)
parser.add_argument('--max-feedback-given', type=int, default=10, help="Number of instances of user feedback (e.g. oracle queries) allowed")
parser.add_argument('--num_seeds', type=int, default=10)

def sample_cluster_seeds(features, labels, num_seed_points_per_label = 1, aggregate="mean"):
    points_by_cluster = defaultdict(list)
    original_index_by_cluster = defaultdict(list)
    for i, (f, l) in enumerate(zip(features, labels)):
        points_by_cluster[l].append(f)
        original_index_by_cluster[l].append(i)
    labels = []
    for label, points in points_by_cluster.items():
        sample = set(random.sample(range(len(points)), k=num_seed_points_per_label))
        for i, point in enumerate(points):
            if i in sample:
                y_value = label
            else:
                y_value = -1
            labels.append(y_value)
    return np.array(labels)

def cluster(semisupervised_algo, features, labels, num_clusters, max_feedback_given=None):
    assert semisupervised_algo in ["KMeans", "PCKMeans", "ConstrainedKMeans", "SeededKMeans"]
    if semisupervised_algo == "KMeans":
        clusterer = KMeans(n_clusters=num_clusters)
        clusterer.fit(features)
    elif semisupervised_algo == "PCKMeans":
        oracle = ExampleOracle(labels, max_queries_cnt=max_feedback_given)

        active_learner = MinMax(n_clusters=num_clusters)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        clusterer = PCKMeans(n_clusters=num_clusters)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    elif semisupervised_algo == "ConstrainedKMeans":
        clusterer = ConstrainedKMeans(n_clusters=num_clusters)
        cluster_seeds = sample_cluster_seeds(features, labels)
        clusterer.fit(features, y=cluster_seeds)
    elif semisupervised_algo == "SeededKMeans":
        clusterer = SeededKMeans(n_clusters=num_clusters)
        cluster_seeds = sample_cluster_seeds(features, labels)
        clusterer.fit(features, y=cluster_seeds) 
    else:
        raise ValueError(f"Algorithm {semisupervised_algo} not supported.")
    return clusterer

def compare_algorithms(features, labels, num_clusters, max_feedback_given=None, algorithms=["KMeans", "PCKMeans", "ConstrainedKMeans", "SeededKMeans"], num_seeds=3):
    algo_results = defaultdict(list)
    for seed in range(num_seeds):
        set_seed(seed)
        for semisupervised_algo in algorithms:
            clusterer = cluster(semisupervised_algo, features, labels, num_clusters, max_feedback_given=max_feedback_given)
            rand_score = metrics.adjusted_rand_score(labels, clusterer.labels_)
            algo_results[semisupervised_algo].append(rand_score)
    return algo_results

if __name__ == "__main__":
    args = parser.parse_args()
    features, labels = load_dataset(args.dataset)
    assert set(labels) == set(range(len(set(labels))))

    algorithms=["KMeans", "PCKMeans", "ConstrainedKMeans", "SeededKMeans"]
    results = compare_algorithms(features, labels, args.num_clusters, max_feedback_given=args.max_feedback_given, num_seeds=args.num_seeds)
    summarized_results = summarize_results(results)
    print(json.dumps(summarized_results, indent=2))