from collections import defaultdict
import random
from sklearn import datasets, metrics

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

def load_dataset(dataset_name, data_path):
    assert dataset_name in ["iris", "20_newsgroups_all", "20_newsgroups_sim3", "OPIEC59k"]
    if dataset_name == "iris":
        samples, gold_cluster_ids = datasets.load_iris(return_X_y=True)
    elif dataset_name == "20_newsgroups_all":
        samples, gold_cluster_ids = preprocess_20_newsgroups()
    if dataset_name == "20_newsgroups_sim3":
        samples, gold_cluster_ids = preprocess_20_newsgroups(topics=["comp.graphics", "comp.os.ms-windows.misc", "comp.windows.x"])
        gold_cluster_ids = reorder_labels(gold_cluster_ids)
    elif dataset_name == "OPIEC59k":
        samples, gold_cluster_ids = [], []
    return samples, gold_cluster_ids