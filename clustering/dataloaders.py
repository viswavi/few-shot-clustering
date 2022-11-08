from collections import defaultdict, namedtuple
import os
import numpy as np
import random
from sklearn import datasets, metrics

import sys
sys.path.append("cmvc")
from CMVC_main_opiec import CMVC_Main
from helper import invertDic

def preprocess_20_newsgroups(per_topic_samples = 100, shuffle=True, topics=None):
    newsgroups = datasets.fetch_20newsgroups(subset='all')
    group_by_topics = defaultdict(list)
    target_to_name = dict(enumerate(newsgroups["target_names"]))
    for target, text in zip(newsgroups['target'], newsgroups['data']):
        if topics is None or target_to_name[target] in topics:
            group_by_topics[target].append(text)
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

def load_dataset(dataset_name, data_path, dataset_split=None):
    assert dataset_name in ["iris", "20_newsgroups_all", "20_newsgroups_sim3", "OPIEC59k"]
    if dataset_name == "iris":
        samples, gold_cluster_ids = datasets.load_iris(return_X_y=True)
        side_information = None
    elif dataset_name == "20_newsgroups_all":
        samples, gold_cluster_ids = preprocess_20_newsgroups()
        side_information = None
    if dataset_name == "20_newsgroups_sim3":
        samples, gold_cluster_ids = preprocess_20_newsgroups(topics=["comp.graphics", "comp.os.ms-windows.misc", "comp.windows.x"])
        gold_cluster_ids = reorder_labels(gold_cluster_ids)
        side_information = None
    elif dataset_name == "OPIEC59k":
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
        out_path = os.path.join(out_dir, dataset_processed_version_name)
        dataset_file = os.path.join(data_path, dataset_name, dataset_name + '_' + dataset_split)
        use_assume = True
        mock_args = MockArgs(dataset_name, file_triples, file_entEmbed, file_relEmbed, file_entClust, file_relClust, file_sideinfo, file_sideinfo_pkl, file_results, out_path, dataset_file, use_assume)
        cmvc = CMVC_Main(mock_args)
        cmvc.get_sideInfo()

        kg_features = np.load(open(os.path.join(data_path, dataset_name, "relation_view_embed.npz"), 'rb'))
        bert_features = np.load(open(os.path.join(data_path, dataset_name, "context_view_embed.npz"), 'rb'))
        cmvc.kg_dimension = kg_features.shape[1]
        cmvc.bert_dimension = bert_features.shape[1]
        samples = np.hstack([kg_features, bert_features])

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

        cluster_id_to_name = invertDic(cluster_name_to_id, 'o2o')
        cmvc.cluster_id_to_name = cluster_id_to_name
        side_information = cmvc

    return samples, gold_cluster_ids, side_information