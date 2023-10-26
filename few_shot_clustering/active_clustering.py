    # python active_clustering.py --dataset iris --num_clusters 3 --num_seeds 10
# python active_clustering.py --dataset 20_newsgroups_all --feature_extractor TFIDF --max-feedback-given 100 --num_clusters 20 --verbose
# python active_clustering.py --dataset 20_newsgroups_sim3 --feature_extractor TFIDF --max-feedback-given 500 --num_clusters 3 --verbose
# python active_clustering.py --dataset 20_newsgroups_diff3 --feature_extractor TFIDF --max-feedback-given 500 --num_clusters 3 --verbose
'''
python active_clustering.py --dataset synthetic_data \
    --num_clusters 5 \
    --num-seeds 5 \
    --plot-clusters \
    --plot-dir /tmp/synthetic_data_vanilla_kmeans_clusters

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


from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans, CardinalityConstrainedPCKMeans, GPTExpansionClustering, KMeansCorrection
from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans
from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.semi_supervised.labeled_data.dec import DEC
# from sklearn.cluster import KMeans
from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.semi_supervised.labeled_data.seededkmeans import SeededKMeans
from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.semi_supervised.labeled_data.constrainedkmeans import ConstrainedKMeans
from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.active.pairwise_constraints import ExampleOracle, GPT3Oracle, construct_pairwise_oracle_single_example, GPT3ComparativeOracle, DistanceBasedSelector, LabelBasedSelector, ExploreConsolidate, MinMax, SimilarityFinder, MinMaxFinetune
from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.active.pairwise_constraints import Random

from few_shot_clustering.cmvc.helper import invertDic
from few_shot_clustering.cmvc.metrics import pairwiseMetric, calcF1
from few_shot_clustering.cmvc.test_performance import cluster_test
from few_shot_clustering.cmvc.model_max_margin import KGEModel
from few_shot_clustering.cmvc.Context_view import BertClassificationModel

from few_shot_clustering.eval_utils import cluster_acc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["iris", "tweet", "clinc", "bank77", "20_newsgroups_all", "20_newsgroups_full", "20_newsgroups_sim3", "20_newsgroups_diff3", "reverb45k", "OPIEC59k", "reverb45k-raw", "OPIEC59k-raw", "OPIEC59k-kg", "OPIEC59k-text", "synthetic_data"], default="iris", help="Clustering dataset to experiment with")
parser.add_argument("--algorithms", action="append")
parser.add_argument('--data-path', type=str, default=None, help="Path to clustering data, if necessary")
parser.add_argument('--dataset-split', type=str, default=None, help="Dataset split to use, if applicable")
parser.add_argument('--num_clusters', type=int, default=3)
parser.add_argument('--pckmeans-w', type=float, default=0.25, help="The 'w' parameter for pairwise constraint k-means")
parser.add_argument('--max-feedback-given', type=int, default=100, help="Number of instances of user feedback (e.g. oracle queries) allowed")
parser.add_argument('--num-corrections', type=int, default=None)
parser.add_argument('--num_seeds', type=int, default=10)
parser.add_argument('--num-reinit', type=int, default=1)
parser.add_argument('--feature_extractor', type=str, choices=["identity", "BERT", "TFIDF"], default="identity")
parser.add_argument('--normalize-vectors', action="store_true", help="Normalize vectors")
parser.add_argument('--split-normalization', action="store_true", help="Normalize per-view components separately (for multi-view clustering)")
parser.add_argument('--init', type=str, choices=["random", "k-means++", "k-means"], default="random", help="Initialization algorithm to use for k-means.")
parser.add_argument('--plot-clusters', action="store_true", help="Whether to plot clusters")
parser.add_argument('--plot-dir', type=str, default=None, help="Directory to store cluster plots")
parser.add_argument('--include-linear-transformation', action="store_true", help="Whether to learn a linear transformation (for the DEC model)")
parser.add_argument('--include-contrastive-loss', action="store_true", help="Whether to include a contrastive loss (for the DEC model)")
parser.add_argument('--tensorboard-dir', type=str, default="tmp", help="Directory name to use for tensorboard")
parser.add_argument('--verbose', action="store_true")



def sample_cluster_seeds(features, labels, max_feedback_given = 0, aggregate="mean"):
    assert len(features) == len(labels)
    labels_list = [-1 for _ in range(len(features))]

    original_index_by_cluster = defaultdict(list)
    for i, (f, l) in enumerate(zip(features, labels)):
        original_index_by_cluster[l].append(i)

    label_values = list(original_index_by_cluster.keys())
    random.shuffle(label_values)

    min_feedback_per_label = max_feedback_given // len(label_values)
    num_labels_with_extra_point = max_feedback_given % len(label_values)
    feedback_per_label = [min_feedback_per_label + int(i < num_labels_with_extra_point) for i in range(len(label_values))]


    feedback_counter = 0
    for i, label in enumerate(label_values):
        num_feedback_for_label = min(len(original_index_by_cluster[label]), feedback_per_label[i])
        labeled_point_indices = random.sample(original_index_by_cluster[label], num_feedback_for_label)
        for point_index in original_index_by_cluster[label]:
            if point_index in labeled_point_indices:
                labels_list[point_index] = label
                feedback_counter += 1

    assert feedback_counter <= max_feedback_given

    return np.array(labels_list)

def construct_pairwise_oracle_prompt(dataset_name, documents, side_information):
    if isinstance(side_information, list):
        side_info = None
    else:
        side_info = side_information.side_info
    if dataset_name == "OPIEC59k":
        instruction = """You are tasked with clustering entity strings based on whether they refer to the same Wikipedia article. To do this, you will be given pairs of entity names and asked if their anchor text, if used separately to link to a Wikipedia article, is likely referring to the same article. Entity names may be truncated, abbreviated, or ambiguous.

To help you make this determination, you will be given up to three context sentences from Wikipedia where the entity is used as anchor text for a hyperlink. Amongst each set of examples for a given entity, the entity for all three sentences is a link to the same article on Wikipedia. Based on these examples, you will decide whether the first entity and the second entity listed would likely link to the same Wikipedia article if used as separate anchor text.

Please note that the context sentences may not be representative of the entity's typical usage, but should aid in resolving the ambiguity of entities that have similar or overlapping meanings.

To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the entities will generally be used in the same contexts, whether the context sentences mention the same topic, and whether the entities have the same domain and scope of meaning.

Your task will be considered successful if the entities are clustered into groups that consistently refer to the same Wikipedia articles."""
        example_1 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["B.A"]], documents[side_info.ent2id["M.D."]], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["B.A"]], documents[side_info.ent2id["bachelor"]], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Duke of York"]], documents[side_info.ent2id["Frederick"]], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Academy Award"]], documents[side_info.ent2id["Best Actor in Supporting Role"]], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    elif dataset_name == "reverb45k":
        instruction = """You are tasked with clustering entity strings based on whether they link to the same entity on the Freebase knowledge graph. To do this, you will be given pairs of entity names and asked if these strings, if linked to a knowledge graph, are likely referring to the same entity (e.g. a concept, person, or organization). Entity names may be truncated, abbreviated, or ambiguous.

To help you make this determination, you will be given up to three context sentences from the internet that mention an entity. Amongst each set of examples for a given entity, assume that the entity mentioned in all three context sentences links refers to the same object. Based on these examples, you will decide whether the first entity and the second entity listed are likely to link to the *same* knowledge graph entity.

Please note that the context sentences may not be representative of the entity's typical usage, but should aid in resolving the ambiguity of entities that have similar or overlapping meanings.

To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the entities will generally be used in the same contexts, whether the entities likely refer to the same person or organization, whether the context sentences mention the same topic, and whether the entities have the same domain and scope of meaning.

Your task will be considered successful if the entities are clustered into groups that consistently link to the same knowledge graph node."""
        example_1 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Hannibal"]], documents[side_info.ent2id["Hannibal Barca"]], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Lutheran Church"]], documents[side_info.ent2id["Church"]], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Grove Art Online"]], documents[side_info.ent2id["Oxford Art Online"]], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Charlie Williams"]], documents[side_info.ent2id["Williams"]], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    elif dataset_name == "tweet":
        instruction = """You are tasked with clustering tweets based on whether they discuss the same topic. To do this, you will be given pairs of (stopword-removed) tweets and asked if they discuss the same topic. To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the tweets explicitly mention the same topic or whether they reflect the same contexts.

Your task will be considered successful if the tweets are clustered into groups that consistently discuss the same topic."""
        example_1 = construct_pairwise_oracle_single_example(documents[0], documents[563], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[4], documents[187], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[2135], documents[1218], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[2471], documents[1218], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    elif dataset_name == "clinc":
        instruction = """You are tasked with clustering queries for a task-oriented dialog system based on whether they express the same general user intent. To do this, you will be given pairs of user queries and asked if they express the same general user need or intent.

Your task will be considered successful if the queries are clustered into groups that consistently express the same general intent."""
        example_1 = construct_pairwise_oracle_single_example(documents[1], documents[2], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[70], documents[700], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[1525], documents[1527], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[1500], documents[1000], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    elif dataset_name == "bank77":
        instruction = """You are tasked with clustering queries for a online banking system based on whether they express the same general user intent. To do this, you will be given pairs of user queries and asked if they express the same general user need or intent.

Your task will be considered successful if the queries are clustered into groups that consistently express the same general intent."""
        example_1 = construct_pairwise_oracle_single_example(documents[0], documents[1], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[1990], documents[2001], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[2010], documents[2001], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[2900], documents[3000], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    else:
        raise NotImplementedError
    return "\n\n".join([instruction, prefix])

def construct_keyphrase_expansion_prompt(dataset_name, documents, side_information):
    if isinstance(side_information, list):
        side_info = None
    else:
        side_info = side_information.side_info
    if dataset_name == "OPIEC59k":
        instruction = """You are tasked with clustering entity strings based on whether they refer to the same Wikipedia article. To do this, you will be given pairs of entity names and asked if their anchor text, if used separately to link to a Wikipedia article, is likely referring to the same article. Entity names may be truncated, abbreviated, or ambiguous.

To help you make this determination, you will be given up to three context sentences from Wikipedia where the entity is used as anchor text for a hyperlink. Amongst each set of examples for a given entity, the entity for all three sentences is a link to the same article on Wikipedia. Based on these examples, you will decide whether the first entity and the second entity listed would likely link to the same Wikipedia article if used as separate anchor text.

Please note that the context sentences may not be representative of the entity's typical usage, but should aid in resolving the ambiguity of entities that have similar or overlapping meanings.

To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the entities will generally be used in the same contexts, whether the context sentences mention the same topic, and whether the entities have the same domain and scope of meaning.

Your task will be considered successful if the entities are clustered into groups that consistently refer to the same Wikipedia articles."""
        example_1 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["B.A"]], documents[side_info.ent2id["M.D."]], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["B.A"]], documents[side_info.ent2id["bachelor"]], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Duke of York"]], documents[side_info.ent2id["Frederick"]], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Academy Award"]], documents[side_info.ent2id["Best Actor in Supporting Role"]], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    elif dataset_name == "reverb45k":
        instruction = """You are tasked with clustering entity strings based on whether they link to the same entity on the Freebase knowledge graph. To do this, you will be given pairs of entity names and asked if these strings, if linked to a knowledge graph, are likely referring to the same entity (e.g. a concept, person, or organization). Entity names may be truncated, abbreviated, or ambiguous.

To help you make this determination, you will be given up to three context sentences from the internet that mention an entity. Amongst each set of examples for a given entity, assume that the entity mentioned in all three context sentences links refers to the same object. Based on these examples, you will decide whether the first entity and the second entity listed are likely to link to the *same* knowledge graph entity.

Please note that the context sentences may not be representative of the entity's typical usage, but should aid in resolving the ambiguity of entities that have similar or overlapping meanings.

To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the entities will generally be used in the same contexts, whether the entities likely refer to the same person or organization, whether the context sentences mention the same topic, and whether the entities have the same domain and scope of meaning.

Your task will be considered successful if the entities are clustered into groups that consistently link to the same knowledge graph node."""
        example_1 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Hannibal"]], documents[side_info.ent2id["Hannibal Barca"]], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Lutheran Church"]], documents[side_info.ent2id["Church"]], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Grove Art Online"]], documents[side_info.ent2id["Oxford Art Online"]], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[side_info.ent2id["Charlie Williams"]], documents[side_info.ent2id["Williams"]], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    elif dataset_name == "tweet":
        instruction = """You are tasked with clustering tweets based on whether they discuss the same topic. To do this, you will be given pairs of (stopword-removed) tweets and asked if they discuss the same topic. To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the tweets explicitly mention the same topic or whether they reflect the same contexts.

Your task will be considered successful if the tweets are clustered into groups that consistently discuss the same topic."""
        example_1 = construct_pairwise_oracle_single_example(documents[0], documents[563], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[4], documents[187], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[2135], documents[1218], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[2471], documents[1218], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    elif dataset_name == "clinc":
        instruction = """You are tasked with clustering queries for a task-oriented dialog system based on whether they express the same general user intent. To do this, you will be given pairs of user queries and asked if they express the same general user need or intent.

Your task will be considered successful if the queries are clustered into groups that consistently express the same general intent."""
        example_1 = construct_pairwise_oracle_single_example(documents[1], documents[2], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[70], documents[700], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[1525], documents[1527], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[1500], documents[1000], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    elif dataset_name == "bank77":
        instruction = """You are tasked with clustering queries for a online banking system based on whether they express the same general user intent. To do this, you will be given pairs of user queries and asked if they express the same general user need or intent.

Your task will be considered successful if the queries are clustered into groups that consistently express the same general intent."""
        example_1 = construct_pairwise_oracle_single_example(documents[0], documents[1], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_2 = construct_pairwise_oracle_single_example(documents[1990], documents[2001], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_3 = construct_pairwise_oracle_single_example(documents[2010], documents[2001], "Yes", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        example_4 = construct_pairwise_oracle_single_example(documents[2900], documents[3000], "No", dataset_name, prompt_suffix = None, text_type = None, add_label=True)
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
    else:
        raise NotImplementedError
    return "\n\n".join([instruction, prefix])


def cluster(semisupervised_algo, features, documents, labels, num_clusters, dataset_name, text_type=None, prompt_suffix=None, num_corrections=None, split=None, init="random", max_feedback_given=None, normalize_vectors=False, split_normalization=False, num_reinit=1, include_linear_transformation=False, include_contrastive_loss=False, verbose=False, side_information=None, process_raw_data=False, pckmeans_w=None, seed=None):
    pairwise_constraint_cache_name = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_pairwise_constraint_cache.jsonl"
    sentence_unprocessing_mapping_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_{split}_sentence_unprocessing_map.json"
    assert semisupervised_algo in ["KMeans", "KMeansCorrection", "GPTExpansionClustering", "GPTPairwiseClustering", "GPTPairwiseClusteringMinMax", "GPTPairwiseClusteringExploreSimilar", "GPTPairwiseClusteringOracleFree", "GPT_CC_PCKMeans", "CardinalityConstrainedPCKMeans", "PCKMeans", "OraclePCKMeans", "ActivePCKMeans", "ActiveFinetunedPCKMeans", "ConstrainedKMeans", "SeededKMeans"]
    if semisupervised_algo == "DEC":
        clusterer = DEC(n_clusters=num_clusters, normalize_vectors=normalize_vectors, split_normalization=split_normalization, verbose=verbose, cluster_init=init, labels=labels, canonicalization_side_information=side_information, include_contrastive_loss=include_contrastive_loss, linear_transformation=include_linear_transformation, tensorboard_parent_dir=tensorboard_parent_dir, tensorboard_dir=tensorboard_dir)
        clusterer.fit(features)
    elif semisupervised_algo == "KMeans":
        clusterer = KMeans(n_clusters=num_clusters, normalize_vectors=normalize_vectors, split_normalization=split_normalization, init=init, num_reinit=num_reinit, verbose=verbose)
        clusterer.fit(features)
    elif semisupervised_algo == "KMeansCorrection":
        labels_cache_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/output/{dataset_name}_kmeans_labels.json"
        cluster_centers_cache_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/output/{dataset_name}_kmeans_cluster_centers.npy"
        if os.path.exists(labels_cache_file):
            cluster_predictions = json.load(open(labels_cache_file))
            cluster_centers = np.load(cluster_centers_cache_file)
        else:
            kmeans_clusterer = KMeans(n_clusters=num_clusters, normalize_vectors=normalize_vectors, split_normalization=split_normalization, init=init, num_reinit=num_reinit, verbose=verbose)
            kmeans_clusterer.fit(features)
            cluster_predictions = kmeans_clusterer.labels_
            cluster_centers = kmeans_clusterer.cluster_centers_
            json.dump([int(l) for l in cluster_predictions], open(labels_cache_file, 'w'))
            np.save(cluster_centers_cache_file, cluster_centers)

        prompt = construct_pairwise_oracle_prompt(dataset_name, documents, side_information)
        oracle = GPT3Oracle(features, prompt, documents, dataset_name=dataset_name, prompt_suffix=prompt_suffix, text_type=text_type, max_queries_cnt=max_feedback_given, cache_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_pairwise_constraint_cache.jsonl")
        clusterer = KMeansCorrection(oracle, cluster_predictions, cluster_centers, labels)
        clusterer.fit(features, num_corrections = num_corrections)

    elif semisupervised_algo == "GPTExpansionClustering":
        cache_file_name = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_gpt_paraphrase_cache.jsonl"
        clusterer = GPTExpansionClustering(features, labels, documents, dataset_name=dataset_name, split=split, n_clusters=num_clusters, side_information=side_information, cache_file_name=cache_file_name)
        clusterer.fit(features, labels)

    elif semisupervised_algo == "GPTPairwiseClustering":
        prompt = construct_pairwise_oracle_prompt(dataset_name, documents, side_information)
        oracle = GPT3Oracle(features, prompt, documents, dataset_name=dataset_name, prompt_suffix=prompt_suffix, text_type=text_type, max_queries_cnt=max_feedback_given, cache_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_pairwise_constraint_cache.jsonl")
        active_learner = LabelBasedSelector(n_clusters=num_clusters)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_
        clusterer = PCKMeans(n_clusters=num_clusters, init=init, normalize_vectors=normalize_vectors, split_normalization=split_normalization, w=pckmeans_w)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        clusterer.constraints_ = pairwise_constraints
        oracle.cache_writer.close()

    elif semisupervised_algo == "GPTPairwiseClusteringOracleFree":
        prompt = construct_pairwise_oracle_prompt(dataset_name, documents, side_information)
        oracle = GPT3Oracle(features, prompt, documents, dataset_name=dataset_name, prompt_suffix=prompt_suffix, text_type=text_type, max_queries_cnt=max_feedback_given, cache_file = f"/projects/ogma1/vijayv/few-shot-clustering/clustering/file/gpt3_cache/{dataset_name}_pairwise_constraint_cache_reprod.jsonl")

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

    elif semisupervised_algo == "GPTPairwiseClusteringMinMax":
        prompt = construct_pairwise_oracle_prompt(dataset_name, documents, side_information)
        oracle = GPT3Oracle(features, prompt, documents, dataset_name=dataset_name, prompt_suffix=prompt_suffix, text_type=text_type, max_queries_cnt=max_feedback_given, cache_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_pairwise_constraint_cache.jsonl")

        print("Collecting Constraints")
        set_seed(0)
        active_learner = MinMax(n_clusters=num_clusters)
        example_oracle = ExampleOracle(labels, max_queries_cnt=max_feedback_given)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        print("Training PCKMeans")
        set_seed(seed)
        clusterer = PCKMeans(n_clusters=num_clusters, init=init, normalize_vectors=True, split_normalization=True, side_information=side_information, w=pckmeans_w)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        clusterer.constraints_ = pairwise_constraints
        if isinstance(oracle, GPT3Oracle) and os.path.exists(oracle.cache_file):
            oracle.cache_writer.close()


    elif semisupervised_algo == "GPTPairwiseClusteringExploreSimilar":
        prompt = construct_pairwise_oracle_prompt(dataset_name, documents, side_information)
        oracle = GPT3Oracle(features, prompt, documents, dataset_name=dataset_name, prompt_suffix=prompt_suffix, text_type=text_type, max_queries_cnt=max_feedback_given, cache_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_pairwise_constraint_cache.jsonl")

        print("Collecting Constraints")
        set_seed(0)
        active_learner = SimilarityFinder(n_clusters=num_clusters)
        #example_oracle = ExampleOracle(labels, max_queries_cnt=1000000)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_
        _ = '''
        example_oracle = ExampleOracle(labels, max_queries_cnt=1000000)
        ml_oracle = [example_oracle.query(i, j) for i, j in pairwise_constraints[0]]
        cl_oracle = [example_oracle.query(i, j) for i, j in pairwise_constraints[1]]
        '''

        print("Training PCKMeans")
        set_seed(seed)
        clusterer = PCKMeans(n_clusters=num_clusters, init=init, normalize_vectors=True, split_normalization=True, side_information=side_information, w=pckmeans_w)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        clusterer.constraints_ = pairwise_constraints
        if isinstance(oracle, GPT3Oracle) and os.path.exists(oracle.cache_file):
            oracle.cache_writer.close()

    elif semisupervised_algo == "GPT_CC_PCKMeans":
        prompt = construct_pairwise_oracle_prompt(dataset_name, documents, side_information)
        oracle = GPT3Oracle(features, prompt, documents, dataset_name=dataset_name, prompt_suffix=prompt_suffix, text_type=text_type, max_queries_cnt=max_feedback_given, cache_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_pairwise_constraint_cache.jsonl")

        active_learner = DistanceBasedSelector(n_clusters=num_clusters)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        clusterer = CardinalityConstrainedPCKMeans(n_clusters=num_clusters, init=init, normalize_vectors=True, split_normalization=True, side_information=side_information, w=pckmeans_w)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        clusterer.constraints_ = pairwise_constraints

    elif semisupervised_algo == "CardinalityConstrainedPCKMeans":
        prompt = construct_pairwise_oracle_prompt(dataset_name, documents, side_information)
        oracle = GPT3Oracle(features, prompt, documents, dataset_name=dataset_name, prompt_suffix=prompt_suffix, text_type=text_type, max_queries_cnt=max_feedback_given, cache_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_pairwise_constraint_cache.jsonl")

        active_learner = DistanceBasedSelector(n_clusters=num_clusters)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_


        clusterer = CardinalityConstrainedPCKMeans(n_clusters=num_clusters, init=init, w=pckmeans_w)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        clusterer.constraints_ = pairwise_constraints

    elif semisupervised_algo == "ActivePCKMeans":
        oracle = ExampleOracle(labels, max_queries_cnt=max_feedback_given)

        active_learner = MinMax(n_clusters=num_clusters)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        clusterer = PCKMeans(n_clusters=num_clusters, init=init, normalize_vectors=True, split_normalization=True, side_information=side_information, w=pckmeans_w)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    elif semisupervised_algo == "ActiveFinetunedPCKMeans":
        oracle = ExampleOracle(labels, max_queries_cnt=max_feedback_given)

        initial_clusterer = KMeans(n_clusters=num_clusters, normalize_vectors=normalize_vectors, split_normalization=split_normalization, init=init, num_reinit=num_reinit, max_iter=10, verbose=verbose)
        initial_clusterer.fit(features)

        active_learner = MinMaxFinetune(n_clusters=num_clusters)
        active_learner.set_initial_clusterer(initial_clusterer)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        clusterer = PCKMeans(n_clusters=num_clusters, init=init, normalize_vectors=True, split_normalization=True, side_information=side_information, w=pckmeans_w)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    elif semisupervised_algo == "PCKMeans":
        oracle = ExampleOracle(labels, max_queries_cnt=max_feedback_given)
        prompt = construct_pairwise_oracle_prompt(dataset_name, documents, side_information)
        gpt3_oracle = GPT3Oracle(features, prompt, documents, dataset_name=dataset_name, prompt_suffix=prompt_suffix, text_type=text_type, max_queries_cnt=max_feedback_given, cache_file = f"/projects/ogma2/users/vijayv/extra_storage/okb-canonicalization/clustering/file/gpt3_cache/{dataset_name}_pairwise_constraint_cache.jsonl")

        oracle.selected_sentences = gpt3_oracle.selected_sentences

        active_learner = DistanceBasedSelector(n_clusters=num_clusters)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        print("Initializing PCKMeans")
        clusterer = PCKMeans(n_clusters=num_clusters, init=init, normalize_vectors=True, split_normalization=True, side_information=side_information, w=pckmeans_w)
        print("Running PCKMeans")
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        clusterer.constraints_ = pairwise_constraints

    elif semisupervised_algo == "OraclePCKMeans":
        oracle = ExampleOracle(labels, max_queries_cnt=max_feedback_given)

        active_learner = LabelBasedSelector(n_clusters=num_clusters)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_
        clusterer = PCKMeans(n_clusters=num_clusters)
        clusterer.fit(features, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        clusterer.constraints_ = pairwise_constraints
    elif semisupervised_algo == "ConstrainedKMeans":
        clusterer = ConstrainedKMeans(n_clusters=num_clusters, init=init)
        cluster_seeds = sample_cluster_seeds(features, labels, max_feedback_given)
        clusterer.fit(features, y=cluster_seeds)
    elif semisupervised_algo == "SeededKMeans":
        clusterer = SeededKMeans(n_clusters=num_clusters, init=init)
        cluster_seeds = sample_cluster_seeds(features, labels, max_feedback_given)
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

def plot_cluster(features, gt_labels, clusterer_labels, metrics, plot_path, pairwise_constraints=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    colormap = dict(zip(list(set(clusterer_labels)), colors))    

    circles = []
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for feat, gt, cluster in zip(features, gt_labels, clusterer_labels):
        x, y = feat
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        circles.append(plt.Circle((x, y), radius=0.5, color=colormap[cluster], alpha=0.1))
        ax.add_patch(circles[-1])
        ax.text(x, y, str(gt), horizontalalignment='center', verticalalignment='center')

    if pairwise_constraints is not None:
        must_links, cannot_links = pairwise_constraints
        for pml in must_links:
            start_x_ml, start_y_ml = features[pml[0]]
            end_x_ml, end_y_ml = features[pml[1]]
            plt.plot(np.array([start_x_ml, end_x_ml]), np.array([start_y_ml, end_y_ml]), '-', linewidth=1, color="black")
        for pcl in cannot_links:
            start_x_cl, start_y_cl = features[pcl[0]]
            end_x_cl, end_y_cl = features[pcl[1]]
            plt.plot(np.array([start_x_cl, end_x_cl]), np.array([start_y_cl, end_y_cl]), '--', linewidth=1, color="purple")

    ax.set_xlim((min_x-0.5, max_x+0.5))
    ax.set_ylim((min_y-0.5, max_y+0.5))
    nmi = metrics["nmi"]
    rand = metrics["rand_score"]
    ax.set_title(f"Comparing ground truth clusters with true labels\nNMI: {round(nmi, 3)}, Rand: {round(rand, 3)}")
    plt.legend()
    fig.savefig(plot_path)
    print(f"Saved plot to {plot_path}")


def compare_algorithms(features,
                       documents,
                       labels,
                       side_information,
                       num_clusters,
                       dataset_name,
                       num_corrections=None,
                       split=None,
                       max_feedback_given=None,
                       num_reinit=1,
                       algorithms=["KMeans", "PCKMeans", "ConstrainedKMeans", "SeededKMeans"],
                       num_seeds=3,
                       verbose=True,
                       normalize_vectors=False,
                       split_normalization=False,
                       init="random",
                       plot_clusters=False,
                       cluster_plot_dir_prefix=None,
                       dataset=None,
                       include_linear_transformation=False,
                       include_contrastive_loss=False,
                       tensorboard_dir="tmp",
                       process_raw_data=False,
                       pckmeans_w=None):
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

        '''
        if dataset == "synthetic_data":
            features, labels = generate_synthetic_data(n_samples_per_cluster=200, global_seed=seed)
            assert set(y) == set(range(len(set(y))))
        '''

        if verbose:
            print(f"Starting experiments for {i}th seed")
        set_seed(seed)
        for semisupervised_algo in algorithms:
            if verbose:
                print(f"Running {semisupervised_algo} for seed {seed}")
            start_time = time.perf_counter()
            clusterer = cluster(semisupervised_algo, features, documents, labels, num_clusters, dataset_name, num_corrections=num_corrections, split=split, max_feedback_given=max_feedback_given, normalize_vectors=normalize_vectors, split_normalization=split_normalization, init=init, num_reinit=num_reinit, verbose=verbose, side_information=side_information, include_linear_transformation=include_linear_transformation, include_contrastive_loss=include_contrastive_loss, tensorboard_dir=tensorboard_dir, process_raw_data=process_raw_data, pckmeans_w=pckmeans_w, seed=seed)
            elapsed_time = time.perf_counter() - start_time
            if verbose:
                print(f"Took {round(elapsed_time, 3)} seconds to cluster points.")
            
            # np.save(open("/projects/ogma1/vijayv/okb-canonicalization/clustering/output/OPIEC59k_test_1/OPIEC59k_clusters/kmeans/cluster_centers.npy", 'wb'), clusterer.cluster_centers_)
            # 
            # breakpoint()
            metric_dict = {}
            algo_results[semisupervised_algo].append(metric_dict)
            if dataset_name.split('-')[0] == "OPIEC59k" or dataset_name.split('-')[0] == "reverb45k":
                optimal_results = cluster_test(side_information.p, side_information.side_info, labels, side_information.true_ent2clust, side_information.true_clust2ent)
                _, _, _, _, _, _, _, _, _, optimal_macro_f1, optimal_micro_f1, optimal_pairwise_f1, _, _, _, _ \
                    = optimal_results
                ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, macro_f1, micro_f1, pairwise_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons  = cluster_test(side_information.p, side_information.side_info, clusterer.labels_, side_information.true_ent2clust, side_information.true_clust2ent)

                # Compute Macro/Macro/Pairwise F1 on OPIEC59k
                metric_dict["macro_f1"] = macro_f1
                metric_dict["micro_f1"] = micro_f1
                metric_dict["pairwise_f1"] = pairwise_f1
                print(f"metric_dict: {metric_dict}")

            rand_score = metrics.adjusted_rand_score(labels, clusterer.labels_)
            metric_dict["rand"] = rand_score
            nmi = metrics.normalized_mutual_info_score(labels, clusterer.labels_)
            metric_dict["nmi"] = nmi

            acc = cluster_acc(np.array(clusterer.labels_), np.array(labels))
            metric_dict["acc"] = acc

            _, pred_clust2ele = generate_cluster_dicts(clusterer.labels_)
            gt_ele2clust, gt_clust2ent = generate_cluster_dicts(labels)
            pair_prec, pair_recall = pairwiseMetric(pred_clust2ele, gt_ele2clust, gt_clust2ent)
            metric_dict["general_pairwise_f1"] = calcF1(pair_prec, pair_recall)

            if plot_clusters:
                cluster_plot_dir = cluster_plot_dir_prefix + "_".join(semisupervised_algo.split())
                os.makedirs(cluster_plot_dir, exist_ok=True)

                clustering_plot_path = os.path.join(cluster_plot_dir, f"{seed}.jpg")
                pcs = None if not hasattr(clusterer, "constraints_") else clusterer.constraints_
                plot_cluster(features, labels, clusterer.labels_, {"rand_score": rand_score, "nmi": nmi}, clustering_plot_path, pairwise_constraints=pcs)

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
    algorithms=args.algorithms
    X, y, documents, side_information = load_dataset(args.dataset, args.data_path, args.dataset_split)
    # assert set(y) == set(range(len(set(y)))), breakpoint()
    features = extract_features(X, args.feature_extractor, args.verbose)
    #algorithms=["KMeans", "ActivePCKMeans", "PCKMeans", "ConstrainedKMeans", "SeededKMeans"]
    process_raw_data = args.dataset.endswith("-raw")
    results = compare_algorithms(features,
                                 documents,
                                 y,
                                 side_information,
                                 args.num_clusters,
                                 args.dataset,
                                 num_corrections=args.num_corrections,
                                 split=args.dataset_split,
                                 max_feedback_given=args.max_feedback_given,
                                 num_seeds=args.num_seeds,
                                 verbose=args.verbose,
                                 normalize_vectors=args.normalize_vectors,
                                 split_normalization = args.split_normalization,
                                 algorithms=algorithms,
                                 init=args.init,
                                 num_reinit=args.num_reinit,
                                 plot_clusters=args.plot_clusters,
                                 cluster_plot_dir_prefix=args.plot_dir,
                                 dataset = args.dataset,
                                 include_contrastive_loss=args.include_contrastive_loss,
                                 include_linear_transformation=args.include_linear_transformation,
                                 tensorboard_dir=args.tensorboard_dir,
                                 process_raw_data=process_raw_data,
                                 pckmeans_w = args.pckmeans_w)
    summarized_results = summarize_results(results)
    print(json.dumps(summarized_results, indent=2))
