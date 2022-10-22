from sklearn import datasets, metrics

def load_dataset(dataset_name):
    assert dataset_name in ["iris", "20_newsgroups"]
    if dataset_name == "iris":
        samples, gold_cluster_ids = datasets.load_iris(return_X_y=True)
    elif dataset_name == "20_newsgroups":
        raise NotImplementedError
    return samples, gold_cluster_ids