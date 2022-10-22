from collections import defaultdict
import numpy as np
import random

def set_seed(seed):
    # set seed for all possible avenues of stochasticity
    np.random.seed(seed=seed)
    random.seed(seed)

def summarize_results(all_seeds_results):
    summarized_results = defaultdict(dict)
    for algo, results in all_seeds_results.items():
        summarized_results[algo]["mean"] = np.mean(results)
        summarized_results[algo]["std"] = np.std(results)
    return summarized_results