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



def GPTPairwiseClusteringOracleFree(features, labels):
    # avoid needing other parameters as used in active_clustering.py
    gpt3_oracle = GPT3Oracle(features, labels, dataset_name, split=split, max_queries_cnt=max_feedback_given, side_information=side_information, read_only=True)
    oracle = ExampleOracle(labels, max_queries_cnt=max_feedback_given)
    oracle.selected_sentences = gpt3_oracle.selected_sentences

    print("Collecting Constraints")
    active_learner = DistanceBasedSelector(n_clusters=num_clusters)
    active_learner.fit(features, oracle=oracle)
    pairwise_constraints = active_learner.pairwise_constraints_

    print("Training PCKMeans")
    clusterer = PCKMeans(n_clusters=num_clusters, init=init, normalize_vectors=True, split_normalization=True, side_information=side_information, w=pckmeans_w)
    clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    clusterer.constraints_ = pairwise_constraints
    if isinstance(oracle, GPT3Oracle) and os.path.exists(oracle.cache_file):
        oracle.cache_writer.close()