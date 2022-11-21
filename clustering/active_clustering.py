# python active_clustering.py --dataset iris --num_clusters 3 --num_seeds 10
# python active_clustering.py --dataset 20_newsgroups_all --feature_extractor TFIDF --max-feedback-given 500 --num_clusters 20 --verbose
# python active_clustering.py --dataset 20_newsgroups_sim3 --feature_extractor TFIDF --max-feedback-given 500 --num_clusters 3 --verbose
# python active_clustering.py --dataset 20_newsgroups_diff3 --feature_extractor TFIDF --max-feedback-given 500 --num_clusters 3 --verbose
'''
python active_clustering.py --dataset OPIEC59k --data-path \
    /projects/ogma1/vijayv/okb-canonicalization/clustering/data \
    --dataset-split test \
    --num_clusters 490 \
    --num_seeds 5

python active_clustering.py --dataset OPIEC59k \
    --data-path /projects/ogma1/vijayv/okb-canonicalization/clustering/data \
    --dataset-split test \
    --num_clusters 490 \
    --num_seeds 1 \
    --normalize-vectors \
    --init k-means++ \
    --verbose |& tee ~/logs/canon/opiec_clustering_simplified_kmeanspp.log
'''

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import argparse
from collections import defaultdict
import json
import numpy as np
import os
import random
import sys
import time

from dataloaders import load_dataset
from experiment_utils import set_seed, summarize_results

sys.path.extend(["..", "."])

if os.getenv("REPO_DIR") is not None:
    sys.path.append(os.path.join(os.getenv("REPO_DIR"), "clustering", "active-semi-supervised-clustering"))
else:
    sys.path.append("active-semi-supervised-clustering")
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans
# from sklearn.cluster import KMeans
from active_semi_clustering.semi_supervised.labeled_data.seededkmeans import SeededKMeans
from active_semi_clustering.semi_supervised.labeled_data.constrainedkmeans import ConstrainedKMeans
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, MinMax

from cmvc.CMVC_main_opiec import CMVC_Main
from cmvc.helper import invertDic
from cmvc.metrics import pairwiseMetric, calcF1
from cmvc.test_performance import cluster_test

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["iris", "20_newsgroups_all", "20_newsgroups_full", "20_newsgroups_sim3", "20_newsgroups_diff3", "OPIEC59k"], default="iris", help="Clustering dataset to experiment with")
parser.add_argument('--data-path', type=str, default=None, help="Path to clustering data, if necessary")
parser.add_argument('--dataset-split', type=str, default=None, help="Dataset split to use, if applicable")
parser.add_argument('--num_clusters', type=int, default=3)
parser.add_argument('--max-feedback-given', type=int, default=10, help="Number of instances of user feedback (e.g. oracle queries) allowed")
parser.add_argument('--num_seeds', type=int, default=10)
parser.add_argument('--num-reinit', type=int, default=1)
parser.add_argument('--feature_extractor', type=str, choices=["identity", "BERT", "TFIDF"], default="identity")
parser.add_argument('--normalize-vectors', action="store_true", help="Normalize vectors")
parser.add_argument('--split-normalization', action="store_true", help="Normalize per-view components separately (for multi-view clustering)")
parser.add_argument('--init', type=str, choices=["random", "k-means++"], default="random", help="Initialization algorithm to use for k-means.")
parser.add_argument('--verbose', action="store_true")


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

def cluster(semisupervised_algo, features, labels, num_clusters, init="random", max_feedback_given=None, normalize_vectors=False, split_normalization=False, num_reinit=1, verbose=False):
    assert semisupervised_algo in ["KMeans", "PCKMeans", "ConstrainedKMeans", "SeededKMeans"]
    if semisupervised_algo == "KMeans":
        clusterer = KMeans(n_clusters=num_clusters, normalize_vectors=False, split_normalization=False, init=init, num_reinit=num_reinit, verbose=verbose)
        clusterer.fit(features)
    elif semisupervised_algo == "PCKMeans":
        oracle = ExampleOracle(labels, max_queries_cnt=max_feedback_given)

        active_learner = MinMax(n_clusters=num_clusters)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        clusterer = PCKMeans(n_clusters=num_clusters)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    elif semisupervised_algo == "ConstrainedKMeans":
        clusterer = ConstrainedKMeans(n_clusters=num_clusters, init=init)
        cluster_seeds = sample_cluster_seeds(features, labels)
        clusterer.fit(features, y=cluster_seeds)
    elif semisupervised_algo == "SeededKMeans":
        clusterer = SeededKMeans(n_clusters=num_clusters, init=init)
        cluster_seeds = sample_cluster_seeds(features, labels)
        clusterer.fit(features, y=cluster_seeds) 
    else:
        raise ValueError(f"Algorithm {semisupervised_algo} not supported.")
    return clusterer

def generate_cluster_dicts(cluster_label_list):
    clust2ele = {}
    for i, cluster_label in enumerate(cluster_label_list):
        if cluster_label not in clust2ele:
            clust2ele[cluster_label] = set()
        clust2ele[cluster_label].add(i)

    ele2clust = invertDic(clust2ele, 'm2os')
    return ele2clust, clust2ele

def compare_algorithms(features, labels, side_information, num_clusters, dataset_name, max_feedback_given=None, num_reinit=1, algorithms=["KMeans", "PCKMeans", "ConstrainedKMeans", "SeededKMeans"], num_seeds=3, verbose=True, normalize_vectors=False, split_normalization=False, init="random"):
    algo_results = defaultdict(list)
    timer = time.perf_counter()

    if normalize_vectors:
        if verbose:
            print(f"Starting feature normalization.")
        if split_normalization:
            timer = time.perf_counter()
            kg_features = normalize(features[:, :300], axis=1, norm="l2")
            bert_features = normalize(features[:, 300:], axis=1, norm="l2")
            features = np.hstack([kg_features, bert_features])
        else:
            features = normalize(features, axis=1, norm="l2")
        if verbose:
            print(f"Feature normalization took {round(time.perf_counter() - timer, 3)} seconds.")

    if verbose:
        print(f"Starting comparison of {num_seeds} seeds:")
    for i, seed in enumerate(range(num_seeds)):
        if verbose:
            print(f"Starting experiments for {i}th seed")
        set_seed(seed)
        for semisupervised_algo in algorithms:
            if verbose:
                print(f"Running {semisupervised_algo} for seed {seed}")
            start_time = time.perf_counter()
            clusterer = cluster(semisupervised_algo, features, labels, num_clusters, max_feedback_given=max_feedback_given, normalize_vectors=normalize_vectors, split_normalization=split_normalization, init=init, num_reinit=num_reinit, verbose=verbose)
            elapsed_time = time.perf_counter() - start_time
            if verbose:
                print(f"Took {round(elapsed_time, 3)} seconds to cluster points.")
            metric_dict = {}
            algo_results[semisupervised_algo].append(metric_dict)
            if dataset_name == "OPIEC59k":
                ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                pair_recall, macro_f1, micro_f1, pairwise_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                    = cluster_test(side_information.p, side_information.side_info, clusterer.labels_, side_information.true_ent2clust, side_information.true_clust2ent)

                # Compute Macro/Macro/Pairwise F1 on OPIEC59k
                metric_dict["macro_f1"] = macro_f1
                metric_dict["micro_f1"] = micro_f1
                metric_dict["pairwise_f1"] = pairwise_f1

            rand_score = metrics.adjusted_rand_score(labels, clusterer.labels_)
            metric_dict["rand"] = rand_score
            nmi = metrics.normalized_mutual_info_score(labels, clusterer.labels_)
            metric_dict["nmi"] = nmi

            _, pred_clust2ele = generate_cluster_dicts(clusterer.labels_)
            gt_ele2clust, gt_clust2ent = generate_cluster_dicts(labels)
            pair_prec, pair_recall = pairwiseMetric(pred_clust2ele, gt_ele2clust, gt_clust2ent)
            metric_dict["general_pairwise_f1"] = calcF1(pair_prec, pair_recall)

        if verbose:
            print("\n")
    return algo_results

def extract_features(dataset, feature_extractor, verbose=False):
    assert feature_extractor in ["identity", "BERT", "TFIDF"]
    if feature_extractor == "identity":
        return dataset
    elif feature_extractor == "TFIDF":
        vectorizer = TfidfVectorizer(max_features=100000, min_df=5, encoding='latin-1', stop_words='english', lowercase=True)
        matrix = np.array(vectorizer.fit_transform(dataset).todense())
        if verbose:
            print(f"Dataset dimensions: {matrix.shape}")
        return matrix
    elif feature_extractor == "BERT":
        raise NotImplementedError


if __name__ == "__main__":
    args = parser.parse_args()
    X, y, side_information = load_dataset(args.dataset, args.data_path, args.dataset_split)
    assert set(y) == set(range(len(set(y))))
    features = extract_features(X, args.feature_extractor, args.verbose)
    #algorithms=["KMeans", "PCKMeans", "ConstrainedKMeans", "SeededKMeans"]
    algorithms=["KMeans"]
    results = compare_algorithms(features,
                                 y,
                                 side_information,
                                 args.num_clusters,
                                 args.dataset,
                                 max_feedback_given=args.max_feedback_given,
                                 num_seeds=args.num_seeds,
                                 verbose=args.verbose,
                                 normalize_vectors=args.normalize_vectors,
                                 split_normalization = args.split_normalization,
                                 algorithms=algorithms,
                                 init=args.init,
                                 num_reinit=args.num_reinit)
    summarized_results = summarize_results(results)
    print(json.dumps(summarized_results, indent=2))