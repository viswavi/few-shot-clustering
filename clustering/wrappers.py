from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import argparse
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import sys
import time
import torch

from dataloaders import load_dataset, generate_synthetic_data
from experiment_utils import set_seed, summarize_results

sys.path.extend(["..", "."])

if os.getenv("REPO_DIR") is not None:
    sys.path.append(os.path.join(os.getenv("REPO_DIR"), "clustering", "active-semi-supervised-clustering"))
else:
    sys.path.append("active-semi-supervised-clustering")
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans, CardinalityConstrainedPCKMeans, GPTExpansionClustering, SCCL, DeepSCCL, KMeansCorrection
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans
from active_semi_clustering.semi_supervised.labeled_data.dec import DEC
# from sklearn.cluster import KMeans
from active_semi_clustering.semi_supervised.labeled_data.seededkmeans import SeededKMeans
from active_semi_clustering.semi_supervised.labeled_data.constrainedkmeans import ConstrainedKMeans
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, GPT3Oracle, GPT3ComparativeOracle, DistanceBasedSelector, LabelBasedSelector, ExploreConsolidate, MinMax, SimilarityFinder, MinMaxFinetune
from active_semi_clustering.active.pairwise_constraints import Random



def LLMGPTPairwiseClusteringOracleFree(features, documents, num_clusters, prompt, prompt_suffix, text_type, max_feedback_given=10000, kmeans_init="k-means++", pckmeans_w=0.4, cache_file=None):
    # avoid needing other parameters as used in active_clustering.py
    oracle = GPT3Oracle(features, prompt, documents, dataset_name=None, prompt_suffix=prompt_suffix, text_type=text_type, cache_file=cache_file, max_queries_cnt=max_feedback_given)

    print("Collecting Constraints")
    active_learner = DistanceBasedSelector(n_clusters=num_clusters)
    active_learner.fit(features, oracle=oracle)
    pairwise_constraints = active_learner.pairwise_constraints_

    print("Training PCKMeans")
    clusterer = PCKMeans(n_clusters=num_clusters, init=kmeans_init, normalize_vectors=True, split_normalization=True, w=pckmeans_w)
    clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    clusterer.constraints_ = pairwise_constraints
    if isinstance(oracle, GPT3Oracle) and os.path.exists(oracle.cache_file):
        oracle.cache_writer.close()
    return clusterer.labels_, clusterer.constraints_

