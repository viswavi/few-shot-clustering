import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist, pdist, euclidean
from cmvc_utils import cos_sim, normalize

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def HAC_getClusters(dataset, embed, cluster_threshold_real, dim_is_bert=False, ave=True):
    if dim_is_bert:
        embed_dim = 768
    else:
        embed_dim = 300
    dist = pdist(embed, metric='cosine')
    if dataset == 'reverb45k':
        if not np.all(np.isfinite(dist)):
            for i in range(len(dist)):
                if not np.isfinite(dist[i]):
                    dist[i] = 0
    clust_res = linkage(dist, method='complete')
    labels = fcluster(clust_res, t=cluster_threshold_real, criterion='distance') - 1

    clusters = [[] for i in range(max(labels) + 1)]
    for i in range(len(labels)):
        clusters[labels[i]].append(i)

    clusters_center = np.zeros((len(clusters), embed_dim), np.float32)
    for i in range(len(clusters)):
        cluster = clusters[i]
        if ave:
            clusters_center_embed = np.zeros(embed_dim, np.float32)
            for j in cluster:
                embed_ = embed[j]
                clusters_center_embed += embed_
            clusters_center_embed_ = clusters_center_embed / len(cluster)
            clusters_center[i, :] = clusters_center_embed_
        else:
            sim_matrix = np.empty((len(cluster), len(cluster)), np.float32)
            for i in range(len(cluster)):
                for j in range(len(cluster)):
                    if i == j:
                        sim_matrix[i, j] = 1
                    else:
                        sim = cos_sim(embed[i], embed[j])
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
            sim_sum = sim_matrix.sum(axis=1)
            max_num = cluster[int(np.argmax(sim_sum))]
            clusters_center[i, :] = embed[max_num]
    return labels, clusters_center

def elbow_method(curve):
    allCoord = np.vstack((np.arange(len(curve)), curve)).T
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / (((lineVec**2).sum()) ** 0.5)
    vecFromFirst = allCoord - allCoord[0]
    vecToLine = vecFromFirst - np.outer((vecFromFirst * lineVecNorm).sum(axis=1), lineVecNorm)
    return (((vecToLine ** 2).sum(axis=1)) ** 0.5).argmax()

def aic(data, centers, labels):
    ni = np.fmax(
        np.unique(labels, return_counts=True)[1], np.finfo(float).eps
    )
    labelmask = np.zeros((centers.shape[0], data.shape[0]))
    for i in range(centers.shape[0]):
        labelmask[i, labels==i] = 1
    denom = data.shape[1] / (data.shape[0] - centers.shape[0])
    sigma = (cdist(centers, data, metric='sqeuclidean') * labelmask).sum() / denom
    '''
    return ((
        (ni * np.log(ni / data.shape[0]))
        - (0.5 * ni * data.shape[1] * np.log(2*np.pi))
        - (0.5 * ni * np.log(sigmai)) - (0.5 * (ni - centers.shape[0]))
    ).sum() - centers.shape[0])
    '''
    return (-2 *
        (ni * np.log(ni)) - (ni * data.shape[0])
        - (0.5 * ni * data.shape[1]) * np.log(2*np.pi*sigma)
        - ((ni - 1) * data.shape[1] * 0.5)
    ).sum() + (2 * centers.shape[0])
    #'''

def bic(data, centers, labels):
    ni = np.fmax(
        np.unique(labels, return_counts=True)[1], np.finfo(float).eps
    )
    labelmask = np.zeros((centers.shape[0], data.shape[0]))
    for i in range(centers.shape[0]):
        labelmask[i, labels==i] = 1
    '''
    denom = ni - centers.shape[0]
    denom[denom==0] = np.finfo(float).eps
    sigmai = np.fmax(
        (cdist(centers, data, metric='sqeuclidean')
        * labelmask).sum(axis=1)
        / denom
        , np.finfo(float).eps
    )
    '''
    denom = data.shape[1] / (data.shape[0] - centers.shape[0])
    sigma = (cdist(centers, data, metric='sqeuclidean') * labelmask).sum() / denom
    return (
        (ni * np.log(ni)) - (ni * data.shape[0])
        - (0.5 * ni * data.shape[1]) * np.log(2 * np.pi * sigma)
        - ((ni - 1) * data.shape[1] * 0.5)
    ).sum() - (0.5 * centers.shape[0] * np.log(data.shape[0]) * (data.shape[1] + 1))

def calinski_harabasz(data, centers, labels) :
    trB = (
        cdist(
            centers, data.mean(axis=0)[None,:], metric='sqeuclidean'
        ).sum(axis=1)
    ).dot((np.unique(labels, return_counts=True))[1])
    trW = (
        cdist(centers, data, metric='sqeuclidean'
    ).min(axis=0)).sum()
    return (
        ((data.shape[0] - centers.shape[0]) * trB)
        / ((centers.shape[0] - 1) * trW)
    )

def classification_entropy(data, centers) :
    dist = cdist(centers, data, metric='sqeuclidean')
    u = 1 / np.fmax(dist, np.finfo(float).eps)
    u = np.fmax(u / u.sum(axis=0), np.finfo(float).eps)
    return - (u * np.log(u)).sum() / data.shape[0]

def compose_within_between(data, centers, centerskmax):
    k = centers.shape[0]
    n = data.shape[0]

    dist = np.fmax(cdist(centers, data), np.finfo(float).eps)
    u = 1 / dist
    u = u / u.sum(axis=0)
    sigma = np.zeros((k, data.shape[1]))
    for iter1 in range(k):
        sigma[iter1,:] = (
            ((data - centers[iter1,:]) ** 2).T * u[iter1,:]
        ).mean(axis=1)
    sigma_x = ((data - data.mean(axis=0)) ** 2).mean(axis=0)
    Scat = np.linalg.norm(sigma, axis=1).mean() / np.linalg.norm(sigma_x)

    dist_centers = cdist(centers, centers)
    dmax = dist_centers.max()
    np.fill_diagonal(dist_centers, np.inf)
    dmin = dist_centers.min()
    np.fill_diagonal(dist_centers, 0)
    Dis = (dmax / dmin) * (1 / dist_centers.sum(axis=1)).sum()

    dist_centerskmax = cdist(centerskmax, centerskmax)
    dmaxkmax = dist_centerskmax.max()
    np.fill_diagonal(dist_centerskmax, np.inf)
    dminkmax = dist_centerskmax.min()
    np.fill_diagonal(dist_centerskmax, 0)
    alpha = (dmaxkmax / dminkmax) * (1 / dist_centerskmax.sum(axis=1)).sum()

    return alpha*Scat + Dis

def davies_bouldin(data, centers, labels):
    k = centers.shape[0]

    cluster_dists = cdist(
        centers, data, metric='sqeuclidean'
    ).min(axis=0)
    unique_labels, cluster_size = np.unique(labels, return_counts=True)
    cluster_sigma = (
        [cluster_dists[labels==i].sum() for i in unique_labels]
        / cluster_size
    ) ** 0.5

    center_dists = cdist(centers, centers)
    np.fill_diagonal(center_dists, 1)

    return ((
            (cluster_sigma[:,None] + cluster_sigma[None,:])
            / center_dists
        ).max(axis=0)).sum() / k

def dunn(pairwise_distances, labels):
    #pairwise_distances = cdist(data, data)

    inter_center_dists = +np.inf
    intra_center_dists = 0
    for iter1 in range(len(np.unique(labels))):
        inter_center_dists = min(
            inter_center_dists,
            pairwise_distances[labels==iter1,:][:,labels!=iter1].min()
        )
        intra_center_dists = max(
            intra_center_dists,
            pairwise_distances[labels==iter1,:][:,labels==iter1].max()
        )
    return inter_center_dists / np.fmax(intra_center_dists, 1.0e-16)

def fukuyama_sugeno(data, centers, m=2):
    if centers.ndim == 1:
        centers = centers[None,:]
    dist = np.fmax(cdist(centers, data, metric='sqeuclidean'), np.finfo(np.float).eps)
    u = (1 / dist)
    um = (u / u.sum(axis=0)) ** m
    return ((um * dist).sum() - cdist(
            centers, centers.mean(axis=0)[None,:], metric='sqeuclidean'
    ).sum())

def fuzzy_hypervolume(data, centers, m=2) :
    if centers.ndim == 1:
        centers = centers[None,:]
    dist = np.fmax(cdist(centers, data, metric='sqeuclidean'), np.finfo(np.float).eps)
    u = (1 / dist)
    um = (u / u.sum(axis=0)) ** m

    return (((um * dist).sum(axis=1) / um.sum(axis=1)) ** 0.5).sum()

def generate_reference_data(data, B, method='pca'):
    if method == 'uniform':
        reference_data = np.random.uniform(
            low=data.min(axis=0), high=data.max(axis=0),
            size=(B, data.shape[0], data.shape[1])
        )
    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca1 = PCA(n_components=data.shape[1])
        proj_data = pca1.fit_transform(data)
        reference_data_proj = np.random.uniform(
            low=proj_data.min(axis=0), high=proj_data.max(axis=0),
            size=(B, proj_data.shape[0], proj_data.shape[1])
        )
        reference_data = pca1.inverse_transform(reference_data_proj)
    else :
        print('ERROR : Incorrect argument "method"')
        return
    return reference_data


def gap_statistic(data, centers, permuted_data, B=30):
    if centers.ndim == 1:
        centers = centers[None,:]
    k = centers.shape[0]
    wk = cdist(centers, data, metric='sqeuclidean').min(axis=0).sum()
    wk_permuted = np.zeros((B))
    for b in range(B):
        #print(k, permuted_data[b,:,:].shape)
        km1 = KMeans(
            n_clusters=k, n_init=2, max_iter=80, tol=1e-6
        ).fit(permuted_data[b,:,:])
        wk_permuted[b] = -km1.score(permuted_data[b,:,:])
    log_wk_permuted = np.log(wk_permuted)
    return (log_wk_permuted.mean() - np.log(wk)), (((
            ((log_wk_permuted - log_wk_permuted.mean()) ** 2).mean()
    ) ** 0.5) * ((1 + (1 / B)) ** 0.5))

def halkidi_vazirgannis(data, centers, labels):
    var_clust = np.zeros((centers.shape[0], data.shape[1]))
    for i in range(centers.shape[0]):
        var_clust[i] = ((data[labels==i] - centers[i]) ** 2).sum(axis=0) / np.fmax((labels==i).sum(axis=0), np.finfo(float).eps)

    data_clust = (((data - data.mean(axis=0)) ** 2).sum(axis=0)) / data.shape[0]

    scat = np.linalg.norm(var_clust, axis=1).sum() / (centers.shape[0] * np.linalg.norm(data_clust))

    avg_std = ((np.linalg.norm(var_clust, axis=1).sum() ** 0.5)
        / centers.shape[0])

    dens = np.zeros((centers.shape[0], centers.shape[0]))
    for iter1 in range(centers.shape[0]):
        for iter2 in range(centers.shape[0]):
            if iter1 == iter2:
                dens[iter1,iter2] = (
                    cdist(data[labels==iter1], centers[iter1][None,:])
                    <= avg_std
                ).sum()
            else:
                dens[iter1,iter2] = (
                    cdist(
                        data[np.logical_or(labels==iter1, labels==iter2)], ((centers[iter1]+centers[iter2]) * 0.5)[None,:]
                    )
                    <= avg_std
                ).sum()
    for iter1 in range(centers.shape[0]):
        for iter2 in range(centers.shape[0]):
            if iter1 != iter2:
                dens[iter1,iter2] = (
                    dens[iter1,iter2] / np.fmax(
                        max(dens[iter1,iter1], dens[iter2,iter2])
                        , np.finfo(float).eps
                    )
                )
    np.fill_diagonal(dens, 0)
    dens_bw = dens.sum() / (centers.shape[0] * (centers.shape[0] - 1))

    return scat + dens_bw

def hartigan_85(data, centers1, centers2):
    if centers1.ndim == 1:
        return (data.shape[0] - centers1.shape[0] - 1) * ((
            cdist(centers1[None,:], data, metric='sqeuclidean').sum()
            / cdist(centers2, data, metric='sqeuclidean').min(axis=0).sum()
        ) - 1)
    else:
        return (data.shape[0] - centers1.shape[0] - 1) * ((
            cdist(centers1, data, metric='sqeuclidean').min(axis=0).sum()
            / cdist(centers2, data, metric='sqeuclidean').min(axis=0).sum()
        ) - 1)

def I_index(data, centers, p=2) :
    dist = np.fmax(cdist(centers, data), np.finfo(np.float64).eps)
    u = 1 / (dist ** 2)
    u = u / u.sum(axis=0)
    return (
        (
            cdist(data, data.mean(axis=0)[None,:]).sum()
            * cdist(centers, centers).max()
        )
        / (centers.shape[0] * np.sum(u * dist))
    ) ** p

def jump_method(d0, d1, y) :
    return d1 ** (-y) - d0 ** (-y)

def last_leap(all_centers, k_list):
    '''
    The Last Leap:
        Method to identify the number of clusters
        between 1 and sqrt(n_data_points)
    Parameters
    ----------
    all_centers : list or tuple, shape (n_centers)
        All cluster centers from k=2 to k=ceil(sqrt(n_data_points))
    Returns
    -------
    k_est: int
        The estimated number of clusters
    min_dist: array, shape(k_max - 1)
        The index values at each k from k=2 to k=ceil(sqrt(n_data_points))
    '''
    k_min, k_max = min(k_list), max(k_list)
    # k_min, k_max = 2, len(all_centers) + 1

    # min_dist = np.zeros((k_max - 1))
    min_dist = np.zeros((len(k_list)))
    # for i in range(k_min, k_max + 1):
    for i in range(len(k_list)):
        k = k_list[i]
        dist = cdist(
            # all_centers[i - k_min], all_centers[i - k_min],
            all_centers[i], all_centers[i],
            metric='sqeuclidean'
        )
        np.fill_diagonal(dist, +np.inf)
        # min_dist[i - k_min] = dist.min()
        min_dist[i] = dist.min()

    # k_est = (
    #     (min_dist[0:-1] - min_dist[1:]) / min_dist[0:-1]
    # ).argmax() + k_min
    k_est = k_list[((min_dist[0:-1] - min_dist[1:]) / min_dist[0:-1]).argmax()]
    # print('k_est:', k_est)

    # Check for single cluster
    # rest_of_the_data = min_dist[k_est - k_min + 1:]
    # if ((min_dist[k_est - 2] * 0.5) < rest_of_the_data).sum() > 0:
    #     k_est = 1
    # print('k_est:', k_est)

    return k_est, min_dist

def last_leap_origin(all_centers, k_list):
    '''
    The Last Leap:
        Method to identify the number of clusters
        between 1 and sqrt(n_data_points)
    Parameters
    ----------
    all_centers : list or tuple, shape (n_centers)
        All cluster centers from k=2 to k=ceil(sqrt(n_data_points))
    Returns
    -------
    k_est: int
        The estimated number of clusters
    min_dist: array, shape(k_max - 1)
        The index values at each k from k=2 to k=ceil(sqrt(n_data_points))
    '''
    # k_min, k_max = min(k_list), max(k_list)
    k_min, k_max = 2, len(all_centers) + 1

    min_dist = np.zeros((k_max - 1))
    # min_dist = np.zeros((len(k_list)))
    for i in range(k_min, k_max + 1):
    # for i in range(len(k_list)):
    #     k = k_list[i]
        dist = cdist(
            all_centers[i - k_min], all_centers[i - k_min],
            # all_centers[i], all_centers[i],
            metric='sqeuclidean'
        )
        np.fill_diagonal(dist, +np.inf)
        min_dist[i - k_min] = dist.min()
        # min_dist[i] = dist.min()

    k_est = (
        (min_dist[0:-1] - min_dist[1:]) / min_dist[0:-1]
    ).argmax() + k_min
    # k_est = k_list[((min_dist[0:-1] - min_dist[1:]) / min_dist[0:-1]).argmax()]
    # print('k_est:', k_est)

    # Check for single cluster
    rest_of_the_data = min_dist[k_est - k_min + 1:]
    if ((min_dist[k_est - 2] * 0.5) < rest_of_the_data).sum() > 0:
        k_est = 1
    # print('k_est:', k_est)

    return k_est, min_dist


def last_major_leap(all_centers, k_list):
    '''
    The Last Major Leap:
        Method to identify the number of clusters
        between 1 and sqrt(n_data_points)
    Parameters
    ----------
    all_centers : list or tuple, shape (n_centers)
        All cluster centers from k=2 to k=ceil(sqrt(n_data_points))
    Returns
    -------
    k_est: int
        The estimated number of clusters
    min_dist: array, shape(k_max - 1)
        The index values at each k from k=2 to k=ceil(sqrt(n_data_points))
    '''

    # k_min = 2
    # k_max = len(all_centers) + 1
    k_min, k_max = min(k_list), max(k_list)

    # min_dist = np.zeros((k_max - 1))
    min_dist = np.zeros((len(k_list)))
    # for i in range(k_min, k_max + 1):
    for i in range(len(k_list)):
        k = k_list[i]
        dist = cdist(
            # all_centers[i - k_min], all_centers[i - k_min],
            all_centers[i], all_centers[i],
            metric='sqeuclidean'
        )
        np.fill_diagonal(dist, +np.inf)
        # min_dist[i - k_min] = dist.min()
        min_dist[i] = dist.min()

    k_est = 1
    for i in range(min_dist.shape[0] - k_min, 0 - 1, -1):
        if (min_dist[i] * 0.5) > (min_dist[i+1:]).max():
            k_est = i + k_min
            break

    return k_est, min_dist

def last_major_leap_origin(all_centers, k_list):
    '''
    The Last Major Leap:
        Method to identify the number of clusters
        between 1 and sqrt(n_data_points)
    Parameters
    ----------
    all_centers : list or tuple, shape (n_centers)
        All cluster centers from k=2 to k=ceil(sqrt(n_data_points))
    Returns
    -------
    k_est: int
        The estimated number of clusters
    min_dist: array, shape(k_max - 1)
        The index values at each k from k=2 to k=ceil(sqrt(n_data_points))
    '''

    k_min = 2
    k_max = len(all_centers) + 1
    # k_min, k_max = min(k_list), max(k_list)

    min_dist = np.zeros((k_max - 1))
    # min_dist = np.zeros((len(k_list)))
    for i in range(k_min, k_max + 1):
    # for i in range(len(k_list)):
    #     k = k_list[i]
        dist = cdist(
            all_centers[i - k_min], all_centers[i - k_min],
            # all_centers[i], all_centers[i],
            metric='sqeuclidean'
        )
        np.fill_diagonal(dist, +np.inf)
        min_dist[i - k_min] = dist.min()
        # min_dist[i] = dist.min()

    k_est = 1
    for i in range(min_dist.shape[0] - k_min, 0 - 1, -1):
        if (min_dist[i] * 0.5) > (min_dist[i+1:]).max():
            k_est = i + k_min
            break

    return k_est, min_dist

def modified_partition_coefficient(data, centers) :
    dist = np.fmax(cdist(centers, data, metric='sqeuclidean'), np.finfo(np.float64).eps)
    u = (1 / dist)
    um = (u / u.sum(axis=0)) ** 2
    return 1 - (
        (centers.shape[0] / (centers.shape[0] - 1))
        * (1 - (um.sum() / data.shape[0]))
    )

def partition_coefficient(data, centers):
    dist = np.fmax(cdist(centers, data, metric='sqeuclidean'), np.finfo(np.float64).eps)
    u = (1 / dist)
    um = (u / u.sum(axis=0)) ** 2
    return um.sum() / data.shape[0]

def partition_index(data, centers, m=2):
    dist = np.fmax(cdist(centers, data, metric='sqeuclidean'), np.finfo(float).eps)
    u = 1 / dist
    um = (u / u.sum(axis=0)) ** m
    return (
        (um * dist).sum(axis=1)
        / (
            um.sum(axis=1)
            * cdist(centers, centers, metric='sqeuclidean').sum(axis=1)
        )
    ).sum()

def pbmf(data, centers, m=1.5):
    dist = cdist(centers, data)
    u = 1 / np.fmax(dist ** 2, np.finfo(np.float64).eps)
    um = (u / u.sum(axis=0)) ** m
    return (
        (
            cdist(data, data.mean(axis=0)[None,:]).sum()
            * cdist(centers, centers).max()
        )
        / ((um * dist).sum() * centers.shape[0])
    ) ** 2

def pcaes(data, centers):
    dist = np.fmax(cdist(centers, data, metric='sqeuclidean'), np.finfo(np.float64).eps)
    u = 1 / dist
    um = (u / u.sum(axis=0)) ** 2
    dist_centers = cdist(centers, centers, metric='sqeuclidean')
    np.fill_diagonal(dist_centers, +np.inf)
    return (
        (um.sum() / um.sum(axis=1).max()) - (
            np.exp(
                (-dist_centers.min(axis=1) * centers.shape[0])
                / cdist(centers, centers.mean(axis=0)[None,:],
                metric='sqeuclidean').sum()
            )
        ).sum()
    )

def get_crossvalidation_data(data, n_fold=2):
    permuted_data = data[np.random.permutation(data.shape[0]),:]
    xdatas = []
    for iter1 in range(n_fold):
        if iter1 == 0:
            xdatas.append((
                permuted_data[data.shape[0]//n_fold:,:],
                permuted_data[0:data.shape[0]//n_fold,:]
            ))
        elif iter1 == (n_fold-1):
            xdatas.append((
                permuted_data[0:iter1*data.shape[0]//n_fold,:],
                permuted_data[iter1*data.shape[0]//n_fold:,:]
            ))
        else:
            xdatas.append((
                np.vstack((
                    permuted_data[0:iter1*data.shape[0]//n_fold, :],
                    permuted_data[(iter1+1)*data.shape[0]//n_fold:, :]
                )),
                permuted_data[
                    iter1*data.shape[0]//n_fold
                    :(iter1+1)*data.shape[0]//n_fold, :
                ]
            ))
    return xdatas


def prediction_strength(xdatas, n_clusters):
    PS = 0
    for train, test in xdatas:
        print('train:', type(train), len(train), 'test:', type(test), len(test))
        km_train = KMeans(
            n_clusters=n_clusters, max_iter=80, n_init=3, tol=1e-6
        ).fit(train)
        km_test = KMeans(
            n_clusters=n_clusters, max_iter=80, n_init=3, tol=1e-6
        ).fit(test)
        train_labels = cdist(
            km_train.cluster_centers_, test
        ).argmin(axis=0)

        ps_k = +np.inf
        for iterk in range(n_clusters):
            co_occurence = np.outer(
                km_test.labels_==iterk, train_labels==iterk
            )
            np.fill_diagonal(co_occurence, 0)
            ps_k = min(
                ps_k, co_occurence.sum() / (km_test.labels_==iterk).sum()
            )
        PS += ps_k
    return PS / len(xdatas)

def ren_liu_wang_yi(data, centers, labels, m=2) :
    dist = cdist(centers, data, metric='sqeuclidean')
    u = 1 / np.fmax(dist, np.finfo(float).eps)
    um = (u / u.sum(axis=0)) ** m

    return (
        (
            ((um * dist).sum(axis=1) / um.sum(axis=1))
            + (
                cdist(centers.mean(axis=0)[None,:], centers,
                metric='sqeuclidean')
                / centers.shape[0]
            )
        )
        / (
            cdist(centers, centers, metric='sqeuclidean').sum(axis=1)
            / (centers.shape[0] - 1)
        )
    ).sum()

def rezaee(data, centers):
    dist = cdist(centers, data, metric='sqeuclidean')
    u = 1 / np.fmax(dist, np.finfo(float).eps)
    u = u / u.sum(axis=0)

    comp = ((u ** 2) * dist).sum()

    h = -(u * np.log(u)).sum(axis=0)
    k = centers.shape[0]
    sep = 0
    for iter1 in range(k) :
        for iter2 in range(iter1+1,k) :
            if iter1 == iter2:
                continue
            sep = sep + (
                np.minimum(u[iter1,:], u[iter2,:]) * h
            ).sum()
    sep = (4 * sep.sum()) / (k * (k - 1))

    return (sep, comp)

def silhouette(pairwise_distances, labels) :
    k = len(np.unique(labels))
    a = np.zeros((pairwise_distances.shape[0]))
    for iter1 in range(k):
        denom = (labels==iter1).sum()
        if denom == 1:
            denom += np.finfo(float).eps
        a[labels==iter1] = (
            pairwise_distances[labels==iter1,:][:,labels==iter1]
        ).sum(axis=1) / denom
    b = np.zeros((pairwise_distances.shape[0])) + np.inf
    for iter1 in range(k):
        for iter2 in range(k):
            if iter1 == iter2:
                continue
            b[labels==iter1] = np.minimum(b[labels==iter1], (
                (
                    pairwise_distances[labels==iter1,:][:,labels==iter2]
                ).sum(axis=1) / np.fmax((labels==iter2).sum(), np.finfo(float).eps)
            ))
    s = (b - a) / np.maximum(b, a)
    return s.mean()

def slope_statistic(sil, p):
    return -(sil[1:] - sil[0:-1]) * (sil[0:-1] ** p)

def xie_beni(data, centers, m=2):
    dist = np.fmax(cdist(centers, data, metric='sqeuclidean'), np.finfo(np.float64).eps)
    u = 1 / dist
    um = (u / u.sum(axis=0)) ** m
    dist_centers = cdist(centers, centers, metric='sqeuclidean')
    np.fill_diagonal(dist_centers, +np.inf)
    return (um * dist).sum() / (data.shape[0] * dist_centers.min())

def xu_index(data, centers):
    return (
                   data.shape[1] * np.log(
               (
                       cdist(centers, data, metric='sqeuclidean').min(axis=0).sum()
                       / (data.shape[1] * (data.shape[0] ** 2))
               ) ** 0.5
           )
           ) + np.log(centers.shape[0])

def zhao_xu_franti(data, centers, labels):
    return (
        (centers.shape[0] * cdist(centers, data).min(axis=0).sum())
        / (
            np.unique(labels, return_counts=True)[1]
            * cdist(centers, data.mean(axis=0)[None,:])
        ).sum()
    )


class Inverse_JumpsMethod(object):

    def __init__(self, data, k_list, dim_is_bert):
        self.data = data
        self.cluster_list = list(k_list)
        self.dim_is_bert = dim_is_bert
        print('self.cluster_list:', type(self.cluster_list), len(self.cluster_list), self.cluster_list)
        # dimension of 'data'; data.shape[0] would be size of 'data'
        self.p = data.shape[1]

    def Distortions(self, random_state=0):
        # cluster_range = range(1, len(cluster_list) + 1)
        cluster_range = range(0, len(self.cluster_list) + 1)
        """ returns a vector of calculated distortions for each cluster number.
            If the number of clusters is 0, distortion is 0 (SJ, p. 2)
            'cluster_range' -- range of numbers of clusters for KMeans;
            'data' -- n by p array """
        # dummy vector for Distortions
        self.distortions = np.repeat(0, len(cluster_range)).astype(np.float32)
        self.K_list = []
        # for each k in cluster range implement
        for i in tqdm(cluster_range):
            if i == cluster_range[-1]:
                parameter = self.cluster_list[-1] + (self.cluster_list[1] - self.cluster_list[0])
            else:
                parameter = self.cluster_list[i]
            KM = KMeans(n_clusters=parameter, random_state=random_state, n_jobs=20)
            KM.fit(self.data)
            centers = KM.cluster_centers_  # calculate centers of suggested k clusters

            K = parameter
            self.K_list.append(K)
            print('i:', i, 'parameter:', parameter, 'cluster_num:', K)
            # since we need to calculate the mean of mins create dummy vec
            for_mean = np.repeat(0, len(self.data)).astype(np.float32)

            # for each observation (i) in data implement
            for j in range(len(self.data)):
                # dummy for vec of distances between i-th obs and k-center
                dists = np.repeat(0, K).astype(np.float32)

                # for each cluster in KMean clusters implement
                for cluster in range(K):
                    euclidean_d = euclidean(normalize(self.data[j]), normalize(centers[cluster]))
                    dists[cluster] = euclidean_d * euclidean_d / 2
                for_mean[j] = min(dists)

            # take the mean for mins for each observation
            self.distortions[i] = np.mean(for_mean) / self.p
        return self.distortions

    def Jumps(self, distortions=None):
        self.distortions = distortions  # change
        """ returns a vector of jumps for each cluster """

        self.jumps = []
        self.jumps += [np.log(self.distortions[k]) - np.log(self.distortions[k - 1]) \
                       for k in range(1, len(self.distortions))]  # argmax
        print('self.jumps:', type(self.jumps), len(self.jumps), self.jumps)

        # calculate recommended number of clusters
        recommended_index = int(np.argmax(np.array(self.jumps)))

        if recommended_index > 0:
            self.recommended_cluster_number = self.cluster_list[recommended_index-1]
        else:
            self.recommended_cluster_number = int(self.cluster_list[0] - (self.cluster_list[1] - self.cluster_list[0]))
        return self.jumps

class JumpsMethod(object):

    def __init__(self, data):
        self.data = data
        # dimension of 'data'; data.shape[0] would be size of 'data'
        self.p = data.shape[1]
        # vector of variances (1 by p)
        """ 'using squared error rather than Mahalanobis distance' (SJ, p. 12)
        sigmas = np.var(data, axis=0)
        ## by following the authors we assume 0 covariance between p variables (SJ, p. 12)
        # start with zero-matrix (p by p)
        self.Sigma = np.zeros((self.p, self.p), dtype=np.float32)
        # fill the main diagonal with variances for
        np.fill_diagonal(self.Sigma, val=sigmas)
        # calculate the inversed matrix
        self.Sigma_inv = np.linalg.inv(self.Sigma)"""

    def Distortions(self, cluster_list=None, random_state=0):
        """ returns a vector of calculated distortions for each cluster number.
            If the number of clusters is 0, distortion is 0 (SJ, p. 2)
            'cluster_range' -- range of numbers of clusters for KMeans;
            'data' -- n by p array """
        # dummy vector for Distortions
        self.distortions = np.repeat(0, len(cluster_list) + 1).astype(np.float32)
        self.cluster_list = cluster_list

        # for each k in cluster range implement
        # for k in cluster_range:
        for i in range(len(self.cluster_list)):
            # initialize and fit the clusterer giving k in the loop
            k = self.cluster_list[i]

            KM = KMeans(n_clusters=k, random_state=random_state, n_jobs=10)
            KM.fit(self.data)
            # calculate centers of suggested k clusters
            centers = KM.cluster_centers_
            print('i:', i, 'parameter:', k)
            # since we need to calculate the mean of mins create dummy vec
            for_mean = np.repeat(0, len(self.data)).astype(np.float32)

            # for each observation (i) in data implement
            for j in range(len(self.data)):
                # dummy for vec of distances between i-th obs and k-center
                dists = np.repeat(0, k).astype(np.float32)

                # for each cluster in KMean clusters implement
                for cluster in range(k):
                    # calculate the within cluster dispersion
                    tmp = np.transpose(self.data[j] - centers[cluster])
                    """ 'using squared error rather than Mahalanobis distance' (SJ, p. 12)
                    dists[cluster] = tmp.dot(self.Sigma_inv).dot(tmp)"""
                    dists[cluster] = tmp.dot(tmp)

                # take the lowest distance to a class
                for_mean[j] = min(dists)

            # take the mean for mins for each observation
            # self.distortions[k] = np.mean(for_mean) / self.p
            self.distortions[i] = np.mean(for_mean) / self.p

        return self.distortions

    def Jumps(self, Y=None, distortions=None):
        """ returns a vector of jumps for each cluster """
        # if Y is not specified use the one that suggested by the authors (SJ, p. 2)
        if Y is None:
            self.Y = self.p / 2

        else:
            self.Y = Y

        if not distortions is None:
            self.distortions = distortions
        # the first (by convention it is 0) and the second elements
        self.jumps = [0] + [self.distortions[1] ** (-self.Y) - 0]
        self.jumps += [self.distortions[k] ** (-self.Y) \
                       - self.distortions[k - 1] ** (-self.Y) \
                       for k in range(2, len(self.distortions))]
        print('self.jumps:', type(self.jumps), len(self.jumps), self.jumps)

        # calculate recommended number of clusters
        # self.recommended_cluster_number = np.argmax(np.array(self.jumps))
        recommended_index = np.argmax(np.array(self.jumps))

        if recommended_index > 0:
            self.recommended_cluster_number = self.cluster_list[recommended_index-1]
        else:
            self.recommended_cluster_number = int(self.cluster_list[0] - (self.cluster_list[1] - self.cluster_list[0]))

        return self.jumps
