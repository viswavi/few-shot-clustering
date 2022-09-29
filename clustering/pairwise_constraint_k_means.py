# python seeded_k_means.py --dataset iris
# python seeded_k_means.py --dataset 20_newsgroups --data-path data/20_newsgroups/

from sklearn import datasets, metrics

import argparse
import os
import sys

from load_data import load_dataset

sys.path.extend(["..", "."])

if os.getenv("REPO_DIR") is not None:
    sys.path.append(os.path.join(os.getenv("REPO_DIR"), "clustering", "active-semi-supervised-clustering"))
else:
    sys.path.append("active-semi-supervised-clustering")
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, MinMax


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["iris", "20_newsgroups"], default="iris", help="Clustering dataset to experiment with")
parser.add_argument('--data-path', type=str, default=None, help="Path to clustering data, if necessary")
parser.add_argument('--training-file', type=str, required=False, default="data/train_set.jsonl")


if __name__ == "__main__":
    args = parser.parse_args()
    samples, gold_cluster_ids = load_dataset(args.dataset)

    oracle = ExampleOracle(gold_cluster_ids, max_queries_cnt=10)

    active_learner = MinMax(n_clusters=3)
    active_learner.fit(samples, oracle=oracle)
    pairwise_constraints = active_learner.pairwise_constraints_

    clusterer = PCKMeans(n_clusters=3)
    clusterer.fit(samples, ml=pairwise_constraints[0], cl=pairwise_constraints[1])

    metrics.adjusted_rand_score(gold_cluster_ids, clusterer.labels_)