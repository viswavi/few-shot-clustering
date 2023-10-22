from collections import defaultdict
import numpy as np
import random

def set_seed(seed):
    # set seed for all possible avenues of stochasticity
    np.random.seed(seed=seed)
    random.seed(seed)

def summarize_results(all_seeds_results):
    summarized_results = {}
    for algo, results in all_seeds_results.items():
        metric_dict_zipped = {}
        for metrics in results:
            if len(metric_dict_zipped) == 0:
                for metric_name in metrics:
                    metric_dict_zipped[metric_name] = []
            for metric_name, metric_value in metrics.items():
                metric_dict_zipped[metric_name].append(metric_value)
        summarized_results[algo] = {metric_name: {} for metric_name in metric_dict_zipped}
        for metric_name, metric_values in metric_dict_zipped.items():
            summarized_results[algo][metric_name]["mean"] = np.mean(metric_values)
            summarized_results[algo][metric_name]["std"] = np.std(metric_values)
    return summarized_results