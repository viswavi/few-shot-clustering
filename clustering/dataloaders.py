from collections import defaultdict, namedtuple

import os
import numpy as np
import random
from sklearn import datasets, metrics

import sys
sys.path.append("cmvc")
from CMVC_main_opiec import CMVC_Main as CMVC_Main_opiec
from CMVC_main_reverb45k import CMVC_Main as CMVC_Main_reverb
from helper import invertDic

def preprocess_20_newsgroups(per_topic_samples = 100, shuffle=True, topics=None):
    newsgroups = datasets.fetch_20newsgroups(subset='all')
    group_by_topics = defaultdict(list)
    target_to_name = dict(enumerate(newsgroups["target_names"]))
    for target, text in zip(newsgroups['target'], newsgroups['data']):
        if topics is None or target_to_name[target] in topics:
            group_by_topics[target].append(text)
    if per_topic_samples is None:
        all_data = [(text, target) for target in group_by_topics for text in group_by_topics[target]]
    else:
        all_data = [(text, target) for target in group_by_topics for text in group_by_topics[target][:per_topic_samples]]
    if shuffle:
        random.shuffle(all_data)
    text, labels = zip(*all_data)
    return text, labels

def reorder_labels(label_array):
    label_to_new_label_mapping = {}
    for i, old_label in enumerate(list(set(label_array))):
        label_to_new_label_mapping[old_label] = i
    return [label_to_new_label_mapping[l] for l in label_array]

def sample_square_points(lower_left_corner, upper_right_corner, n_points=10, corner_offset=1.0, seed=0):
    x_boundaries = (lower_left_corner[0], upper_right_corner[0])
    y_boundaries = (lower_left_corner[1], upper_right_corner[1])
    
    corner_points = [(lower_left_corner[0] + corner_offset, lower_left_corner[1] + corner_offset),
                     (lower_left_corner[0] + corner_offset, upper_right_corner[1] - corner_offset),
                     (upper_right_corner[0] - corner_offset, upper_right_corner[1] - corner_offset),
                     (upper_right_corner[0] - corner_offset, lower_left_corner[1] + corner_offset),]
    '''

    corner_points = []
    '''

    np.random.seed(seed)
    x_samples = np.random.uniform(x_boundaries[0] + 0.00001, x_boundaries[1], size=n_points - 2)
    y_samples = np.random.uniform(y_boundaries[0] + 0.00001, y_boundaries[1], size=n_points - 2)

    sampled_points = corner_points + list(zip(x_samples, y_samples))
    sampled_points = [list(p) for p in sampled_points]
    return sampled_points

                

def generate_synthetic_data(n_samples_per_cluster, global_seed=2022):
    # 5 squares
    points = []
    labels = []
    square_1 = sample_square_points((0, 0), (5, 5), n_points=n_samples_per_cluster, seed=0)
    points.extend(square_1)
    labels.extend([0 for _ in square_1])
    square_2 = sample_square_points((4, 4), (8, 8), n_points=n_samples_per_cluster, seed=1)
    points.extend(square_2)
    labels.extend([1 for _ in square_2])
    square_3 = sample_square_points((7, 7), (10, 10), n_points=n_samples_per_cluster, seed=2)
    points.extend(square_3)
    labels.extend([2 for _ in square_3])
    square_4 = sample_square_points((7, 1), (11, 5), n_points=n_samples_per_cluster, seed=3)
    points.extend(square_4)
    labels.extend([3 for _ in square_4])
    square_5 = sample_square_points((2, 7), (5, 10), n_points=n_samples_per_cluster, seed=4)
    points.extend(square_5)
    labels.extend([4 for _ in square_5])
    combined_data = list(zip(points, labels))

    np.random.seed(global_seed)
    np.random.shuffle(combined_data)  
    points, labels = zip(*combined_data)
    return np.array(points), labels

def load_dataset(dataset_name, data_path, dataset_split=None):
    assert dataset_name in ["iris", "20_newsgroups_all", "20_newsgroups_full", "20_newsgroups_sim3", "20_newsgroups_diff3", "reverb45k", "OPIEC59k", "reverb45k-raw", "OPIEC59k-raw", "OPIEC59k-kg", "OPIEC59k-text", "synthetic_data"]
    if dataset_name == "iris":
        samples, gold_cluster_ids = datasets.load_iris(return_X_y=True)
        side_information = None
    elif dataset_name == "20_newsgroups_all":
        samples, gold_cluster_ids = preprocess_20_newsgroups(per_topic_samples=100)
        side_information = None
    elif dataset_name == "20_newsgroups_full":
        samples, gold_cluster_ids = preprocess_20_newsgroups(per_topic_samples=None)
        side_information = None
    elif dataset_name == "20_newsgroups_sim3":
        samples, gold_cluster_ids = preprocess_20_newsgroups(topics=["comp.graphics", "comp.os.ms-windows.misc", "comp.windows.x"])
        gold_cluster_ids = reorder_labels(gold_cluster_ids)
        side_information = None
    elif dataset_name == "20_newsgroups_diff3":
        samples, gold_cluster_ids = preprocess_20_newsgroups(topics=["alt.atheism", "rec.sport.baseball", "sci.space"])
        gold_cluster_ids = reorder_labels(gold_cluster_ids)
        side_information = None
    elif dataset_name == "synthetic_data":
        samples, gold_cluster_ids = generate_synthetic_data(n_samples_per_cluster=20)
        side_information = None
    elif dataset_name.split('-')[0] == "OPIEC59k" or dataset_name.split('-')[0] == "reverb45k":
        name_constituents = dataset_name.split("-")
        if len(name_constituents) == 2 and name_constituents[1] in ["kg", "text"]:
            dataset_name = name_constituents[0]
            modality_type = name_constituents[1]
        elif len(name_constituents) == 2 and name_constituents[1] == "raw":
            dataset_name = name_constituents[0]
            modality_type = "all"
        else:
            modality_type = "all"
        # set up OPIEC59k evaluation set
        MockArgs = namedtuple("MockArgs", ["dataset", "file_triples", "file_entEmbed", "file_relEmbed", "file_entClust", "file_relClust", "file_sideinfo", "file_sideinfo_pkl", "file_results", "out_path", "data_path", "use_assume"])
        file_triples = '/triples.txt'  # Location for caching triples
        file_entEmbed = '/embed_ent.pkl'  # Location for caching learned embeddings for noun phrases
        file_relEmbed = '/embed_rel.pkl'  # Location for caching learned embeddings for relation phrases
        file_entClust = '/cluster_ent.txt'  # Location for caching Entity clustering results
        file_relClust = '/cluster_rel.txt'  # Location for caching Relation clustering results
        file_sideinfo = '/side_info.txt'  # Location for caching side information extracted for the KG (for display)
        file_sideinfo_pkl = '/side_info.pkl'  # Location for caching side information extracted for the KG (binary)
        file_results = '/results.json'  # Location for loading hyperparameters

        dataset_processed_version_name = dataset_name + '_' + dataset_split + '_' + '1'
        # This convoluted path operation goes from the path to the dataset's data directory to the path of the output files derived from data
        out_dir = os.path.join(os.path.abspath(os.path.join(data_path, os.pardir)), "output")
        out_path = os.path.join("/projects/ogma2/users/vijayv/extra_storage", dataset_processed_version_name)
        dataset_file = os.path.join(data_path, dataset_name, dataset_name + '_' + dataset_split)
        use_assume = True
        mock_args = MockArgs(dataset_name, file_triples, file_entEmbed, file_relEmbed, file_entClust, file_relClust, file_sideinfo, file_sideinfo_pkl, file_results, out_path, dataset_file, use_assume)
        if dataset_name.split('-')[0] == "OPIEC59k":
            cmvc = CMVC_Main_opiec(mock_args)
        elif dataset_name.split('-')[0] == "reverb45k":
            cmvc = CMVC_Main_reverb(mock_args)

        cmvc.get_sideInfo() 

        kg_features = np.load(open(os.path.join(data_path, dataset_name, "relation_view_embed.npz"), 'rb'))
        bert_features = np.load(open(os.path.join(data_path, dataset_name, "context_view_embed.npz"), 'rb'))
        cmvc.kg_dimension = kg_features.shape[1]
        cmvc.bert_dimension = bert_features.shape[1]
        if modality_type == "kg":
            samples = kg_features
        elif modality_type == "text":
            samples = bert_features
        elif modality_type == "all":
            samples = np.hstack([kg_features, bert_features])
        else:
            raise NotImplementedError

        ent_ids = [cmvc.side_info.ent2id[trp['triple'][0]] for trp in cmvc.side_info.triples]
        cluster_names = [list(cmvc.true_ent2clust[trp['triple_unique'][0]])[0] for trp in cmvc.side_info.triples]
        cluster_name_to_id = {}
        for c in cluster_names:
            if c not in cluster_name_to_id:
                cluster_name_to_id[c] = len(cluster_name_to_id)

        assert set(ent_ids) == set(list(range(max(ent_ids) + 1)))
        gold_cluster_ids = [None for i in range(max(ent_ids) + 1)]
        for i, ent_id in enumerate(ent_ids):
            cluster_name = cluster_names[i]
            gold_cluster_ids[ent_id] = cluster_name_to_id[cluster_name]


        '''
        entity_to_cluster_name = {}
        for i, trp in enumerate(cmvc.side_info.triples):
            ent_name = trp['triple'][0]
            cluster_name = cluster_names[i]
            entity_to_cluster_name[ent_name] = cluster_name
        '''

        cluster_id_to_name = invertDic(cluster_name_to_id, 'o2o')
        cmvc.cluster_id_to_name = cluster_id_to_name
        side_information = cmvc

    return samples, gold_cluster_ids, side_information
