import numpy as np

import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def init_seeded_kmeans_plusplus(X, seed_set, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++. Modified from original at
    https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/cluster/_kmeans.py#L154.

    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    init_clusters_seeds : array, shape N
        Pre-initialized cluster seeds (indices to the dataset) chosen by
        a previous method (such as an oracle). The number of initial cluster
        seeds N must be less than n_clusters.

    n_clusters : integer
        The number of seeds to choose

    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape


    print(f"n_clusters: {n_clusters}")
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    print(f"(BEFORE) centers.shape: {centers.shape}")

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    if seed_set is None or len(seed_set) == 0:
        random_index = random_state.choice(list(range(len(X))))
        seed_set = [random_index]
    else:
        seed_set = seed_set

    # Pick first center randomly
    centers[0] = X[seed_set[0]]

    # Pick first N centers from seeds

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        X[seed_set[0], np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):

        if c < len(seed_set):
            true_cluster_seed = seed_set[c]
            candidate_ids = np.array([true_cluster_seed])
        else:
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            # rand_vals = random_state.random_sample(n_local_trials) * current_pot
            # candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            if len(closest_dist_sq.shape) == 2:
                distances_normalized = closest_dist_sq[0]
            else:
                distances_normalized = closest_dist_sq
            distances_normalized = distances_normalized / sum(distances_normalized)
            candidate_ids = random_state.choice(range(len(distances_normalized)), p=distances_normalized, size=n_local_trials)

            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                    out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
                X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out= distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)


        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
    return centers