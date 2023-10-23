import numpy as np
import pandas as pd
import pickle
import os
from sklearn import datasets  # 导入库
from tqdm import tqdm
from find_k_methods import Inverse_JumpsMethod, JumpsMethod, last_leap, \
    last_major_leap_origin, last_leap_origin, jump_method, I_index, aic, elbow_method, bic, \
    calinski_harabasz, classification_entropy, compose_within_between, davies_bouldin, dunn, fukuyama_sugeno,\
    fuzzy_hypervolume,  halkidi_vazirgannis, \
    modified_partition_coefficient, partition_coefficient, partition_index, pbmf, pcaes, \
    ren_liu_wang_yi, rezaee, silhouette, slope_statistic, xie_beni, xu_index, zhao_xu_franti
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# dataset = 'echocardiogram'
# dataset = 'iris'
# dataset = 'seeds'
dataset = 'wine'
# dataset = 'colon'
# dataset = 'prostate'
# dataset = 'miceprotein' # ok
# dataset = 'abalone'
# dataset = 'cnae-9'
# dataset = 'vowel'
# dataset = 'synthetic_control'

n_jobs=1

print('dataset:', dataset)

if dataset == 'echocardiogram':  # 3
    fname = '../file/real_dataset/echocardiogram.data'
    print('load :', fname)
    df = pd.read_csv(fname, header=None, error_bad_lines=False)
    df = df.to_numpy()
    x_cols, y_cols = np.split(df, [9], axis=1)
    input_data = np.zeros(shape=(106, x_cols.shape[1]))
    labels_list = []
    col_num = 0
    for i in range(df.shape[0]):
        num = 0
        for j in range(11):
            value = df[i][j]
            if value == '?':
                num += 1
        if num == 0:
            input_data[col_num] = x_cols[i]
            labels_list.append(y_cols[i][2])
            col_num += 1
    cluster_num = len(list(set(labels_list)))

elif dataset == 'iris': # 3
    input_data, labels_list = datasets.load_iris(return_X_y=True)
    level_one_min, level_one_max, level_one_gap = 1, 6, 1
    cluster_num = len(list(set(labels_list.tolist())))

elif dataset == 'seeds': # 3
    fname = '../file/real_dataset/seeds_dataset.txt'
    print('load :', fname)
    data_x_and_y = np.loadtxt(fname, delimiter='\t')
    print('data_x_and_y:', type(data_x_and_y), data_x_and_y.shape, data_x_and_y)
    input_data, labels = np.split(data_x_and_y, [7], axis=1)
    labels_list = []
    for label_list in labels.tolist():
        labels_list.append(label_list[0])
    cluster_num = len(list(set(labels_list)))

elif dataset == 'wine': # 3
    input_data, labels_list = datasets.load_wine(return_X_y=True)
    cluster_num = len(list(set(labels_list.tolist())))

elif dataset == 'miceprotein': # 8
    mice = datasets.fetch_openml(name='miceprotein', version=4)
    x_cols, y_cols = mice['data'], mice['target']
    input_data = np.zeros(shape=(552, x_cols.shape[1]))
    labels_list = []
    col_num = 0
    for i in range(x_cols.shape[0]):
        num = 0
        for j in range(x_cols.shape[1]):
            value = x_cols[i][j]
            if np.isnan(value):
                num += 1
        if num == 0:
            input_data[col_num] = x_cols[i]
            labels_list.append(y_cols[i])
            col_num += 1
    labels_list = np.array(labels_list)
    cluster_num = len(list(set(labels_list.tolist())))

elif dataset == 'abalone': # 28
    mice = datasets.fetch_openml(name='abalone')
    x_cols, y_cols = mice['data'], mice['target']
    print('x_cols:', type(x_cols), x_cols.shape, x_cols)
    print('y_cols:', type(y_cols), y_cols.shape, y_cols)
    input_data, labels_list = x_cols, y_cols
    cluster_num = len(list(set(labels_list.tolist())))

elif dataset == 'cnae-9': # 9
    mice = datasets.fetch_openml(name='cnae-9')
    x_cols, y_cols = mice['data'], mice['target']
    print('x_cols:', type(x_cols), x_cols.shape, x_cols)
    print('y_cols:', type(y_cols), y_cols.shape, y_cols)
    input_data, labels_list = x_cols, y_cols
    cluster_num = len(list(set(labels_list.tolist())))

elif dataset == 'vowel': # 10
    mice = datasets.fetch_openml(name='vowel')
    x_cols, y_cols = mice['data'], mice['target']
    print('x_cols:', type(x_cols), x_cols.shape, x_cols)
    print('y_cols:', type(y_cols), y_cols.shape, y_cols)

    input_data, labels_list = x_cols, y_cols
    cluster_num = len(list(set(labels_list.tolist())))

elif dataset == 'synthetic_control': # 10
    mice = datasets.fetch_openml(name='synthetic_control')
    x_cols, y_cols = mice['data'], mice['target']
    print('x_cols:', type(x_cols), x_cols.shape, x_cols)
    print('y_cols:', type(y_cols), y_cols.shape, y_cols)
    input_data, labels_list = x_cols, y_cols
    cluster_num = len(list(set(labels_list.tolist())))

elif dataset == 'colon': # 2
    fname_x = '../file/real_dataset/Colon Cancer/colon.x.txt'
    fname_y = '../file/real_dataset/Colon Cancer/colon.y.txt'
    print('load :', fname_x, fname_y)
    data_x = np.loadtxt(fname_x, delimiter=' ')
    data_y = np.loadtxt(fname_y, delimiter='\t')
    input_data, labels = data_x.transpose(), data_y
    print('input_data:', type(input_data), input_data.shape, input_data)
    print('labels:', type(labels), labels.shape, labels)
    labels_list = []
    for label in labels.tolist():
        labels_list.append(label)
    cluster_num = len(list(set(labels_list)))

elif dataset == 'prostate': # 2
    fname_x = '../file/real_dataset/Prostate Cancer/prostate.x.txt'
    fname_y = '../file/real_dataset/Prostate Cancer/prostate.y.txt'
    print('load :', fname_x, fname_y)
    data_x = np.loadtxt(fname_x, delimiter=' ')
    data_y = np.loadtxt(fname_y, delimiter='\t')
    input_data, labels = data_x.transpose(), data_y
    print('input_data:', type(input_data), input_data.shape, input_data)
    print('labels:', type(labels), labels.shape, labels)
    labels_list = []
    for label in labels.tolist():
        labels_list.append(label)
    cluster_num = len(list(set(labels_list)))

else:
    input_data, labels_list = datasets.load_digits(return_X_y=True) # 10
    cluster_num = len(list(set(labels_list.tolist())))

print('input_data:', type(input_data), input_data.shape)
print('labels_list:', type(labels_list), len(labels_list))
print('cluster_num:', cluster_num)
input_embed = input_data
print('input_embed:', type(input_embed), input_embed.shape)
level_one_min, level_one_max, level_one_gap = 1, int(np.floor(input_data.shape[0] ** 0.5) / 2), 1
cluster_list = range(level_one_min, level_one_max, level_one_gap)
print('level_one_min, level_one_max, level_one_gap:', level_one_min, level_one_max, level_one_gap)
k_list = list(cluster_list)

method2first_cluster_num_dict = dict()
jm = Inverse_JumpsMethod(data=input_embed, k_list=cluster_list, dim_is_bert=None)

jm.Distortions(random_state=0)
distortions = jm.distortions
jm.Jumps(distortions=distortions)
jumps = jm.jumps

level_one_Inverse_JumpsMethod = jm.recommended_cluster_number

print('Inverse_JumpsMethod k:', level_one_Inverse_JumpsMethod)
method2first_cluster_num_dict['Inverse_JumpsMethod'] = level_one_Inverse_JumpsMethod

jm = JumpsMethod(data=input_embed)
print('number of dimensions:', jm.p)
jm.Distortions(random_state=0, cluster_list=cluster_list)
distortions = jm.distortions
jm.Jumps()
level_one_JumpsMethod = jm.recommended_cluster_number
print('level_one_JumpsMethod k:', level_one_JumpsMethod)
method2first_cluster_num_dict['JumpsMethod'] = level_one_JumpsMethod

k_min = level_one_min
k_max = level_one_max
print('k_min:', k_min, 'k_max:', k_max)
print('k_list:', type(k_list), len(k_list), k_list)

all_centers = []
index_I_index = np.zeros((len(k_list)))
index_aic = np.zeros((len(k_list)))
index_elbow = np.zeros((len(k_list)))
index_bic = np.zeros((len(k_list)))
index_calinski_harabasz = np.zeros((len(k_list)))
index_classification_entropy = np.zeros((len(k_list)))
index_compose_within_between = np.zeros((len(k_list)))
index_davies_bouldin = np.zeros((len(k_list)))
index_dunn = np.zeros((len(k_list)))
index_fukuyama_sugeno = np.zeros((len(k_list)))
index_fuzzy_hypervolume = np.zeros((len(k_list)))
index_halkidi_vazirgannis = np.zeros((len(k_list)))
index_modified_partition_coefficient = np.zeros((len(k_list)))
index_partition_coefficient = np.zeros((len(k_list)))
index_partition_index = np.zeros((len(k_list)))
index_pbmf = np.zeros((len(k_list)))
index_pcaes = np.zeros((len(k_list)))
index_prediction_strength = np.zeros((len(k_list)))
index_ren_liu_wang_yi = np.zeros((len(k_list)))
sep_rezaee = np.zeros((len(k_list)))
comp_rezaee = np.zeros((len(k_list)))
index_silhouette = np.zeros((len(k_list)))
index_xie_beni = np.zeros((len(k_list)))
index_xu_index = np.zeros((len(k_list)))
index_zhao_xu_franti = np.zeros((len(k_list)))

km1 = KMeans(n_clusters=k_max, n_init=30, max_iter=300, tol=1e-9, algorithm='auto', n_jobs=n_jobs).fit(input_embed)
centerskmax = np.array(km1.cluster_centers_)
pairwise_distances = cdist(input_embed, input_embed)

for i in tqdm(range(len(k_list))):
    k = k_list[i]
    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(input_embed)
    if k != 1:
        index_I_index[i] = I_index(input_embed, km1.cluster_centers_)
        index_aic[i] = aic(input_embed, km1.cluster_centers_, km1.labels_)
        index_bic[i] = bic(input_embed, km1.cluster_centers_, km1.labels_)
        index_calinski_harabasz[i] = calinski_harabasz(input_embed, km1.cluster_centers_, km1.labels_)
        index_classification_entropy[i] = classification_entropy(input_embed, km1.cluster_centers_)
        if k != k_max - 1:
            index_compose_within_between[i] = compose_within_between(input_embed, km1.cluster_centers_,
                                                                             centerskmax)
        index_davies_bouldin[i] = davies_bouldin(input_embed, km1.cluster_centers_, km1.labels_)
        index_dunn[i] = dunn(pairwise_distances, km1.labels_)
        index_halkidi_vazirgannis[i] = halkidi_vazirgannis(input_embed, km1.cluster_centers_, km1.labels_)
        all_centers.append(km1.cluster_centers_)
        index_modified_partition_coefficient[i] = modified_partition_coefficient(input_embed, km1.cluster_centers_)
        index_partition_coefficient[i] = partition_coefficient(input_embed, km1.cluster_centers_)
        index_partition_index[i] = partition_index(input_embed, km1.cluster_centers_)
        index_pbmf[i] = pbmf(input_embed, km1.cluster_centers_)
        index_pcaes[i] = pcaes(input_embed, km1.cluster_centers_)
        index_ren_liu_wang_yi[i] = ren_liu_wang_yi(input_embed, km1.cluster_centers_, km1.labels_)
        sep_rezaee[i], comp_rezaee[i] = rezaee(input_embed, km1.cluster_centers_)
        index_silhouette[i] = silhouette(pairwise_distances, km1.labels_)
        index_xie_beni[i] = xie_beni(input_embed, km1.cluster_centers_)
        index_xu_index[i] = xu_index(input_embed, km1.cluster_centers_)
        index_zhao_xu_franti[i] = zhao_xu_franti(input_embed, km1.cluster_centers_, km1.labels_)

    index_elbow[i] = -km1.score(input_embed)
    index_fukuyama_sugeno[i] = fukuyama_sugeno(input_embed, km1.cluster_centers_)
    index_fuzzy_hypervolume[i] = fuzzy_hypervolume(input_embed, km1.cluster_centers_)

print('Inverse_JumpsMethod k:', level_one_Inverse_JumpsMethod)
print('JumpsMethod k:', level_one_JumpsMethod)
est_k_aic = k_list[elbow_method(index_aic)]
print('For aic : Selected k =', est_k_aic)
method2first_cluster_num_dict['aic'] = est_k_aic

est_k_bic = k_list[index_bic.argmax()]
print('For bic : Selected k =', est_k_bic)
method2first_cluster_num_dict['bic'] = est_k_bic

est_k_calinski_harabasz = k_list[index_calinski_harabasz.argmax()]
print('For calinski_harabasz : elected k =', est_k_calinski_harabasz)
method2first_cluster_num_dict['calinski_harabasz'] = est_k_calinski_harabasz

est_k_classification_entropy = k_list[index_classification_entropy.argmin()]
print('For classification_entropy : Selected k =', est_k_classification_entropy)
method2first_cluster_num_dict['classification_entropy'] = est_k_classification_entropy

index_compose_within_between[len(k_list)-1] = compose_within_between(input_embed, centerskmax, centerskmax)
est_k_compose_within_between = k_list[index_compose_within_between.argmin()]
print('For compose_within_between : Selected k =', est_k_compose_within_between)
method2first_cluster_num_dict['compose_within_between'] = est_k_compose_within_between

est_k_davies_bouldin = k_list[elbow_method(index_davies_bouldin)]
print('For davies_bouldin : Selected k =', est_k_davies_bouldin)
method2first_cluster_num_dict['davies_bouldin'] = est_k_davies_bouldin

est_k_dunn = k_list[index_dunn.argmax()]
print('For dunn : Selected k =', est_k_dunn)
method2first_cluster_num_dict['dunn'] = est_k_dunn

est_k_elbow = k_list[elbow_method(index_elbow)]
print('For elbow : Selected k =', est_k_elbow)
method2first_cluster_num_dict['elbow'] = est_k_elbow

est_k_fukuyama_sugeno = k_list[elbow_method(index_fukuyama_sugeno)]
print('For fukuyama_sugeno : Selected k =', est_k_fukuyama_sugeno)
method2first_cluster_num_dict['fukuyama_sugeno'] = est_k_fukuyama_sugeno

est_k_fuzzy_hypervolume = k_list[index_fuzzy_hypervolume.argmin()]
print('For fuzzy_hypervolume : Selected k =', est_k_fuzzy_hypervolume)
method2first_cluster_num_dict['fuzzy_hypervolume'] = est_k_fuzzy_hypervolume

est_k_halkidi_vazirgannis = k_list[index_halkidi_vazirgannis.argmin()]
print('For halkidi_vazirgannis : Selected k =', est_k_halkidi_vazirgannis)
method2first_cluster_num_dict['halkidi_vazirgannis'] = est_k_halkidi_vazirgannis

est_k_I_index = k_list[index_I_index.argmax()]
print('For I_index : Selected k =', est_k_I_index)
method2first_cluster_num_dict['I_index'] = est_k_I_index

est_k_partition_coefficient = k_list[index_partition_coefficient.argmax()]
print('For partition_coefficient : Selected k =', est_k_partition_coefficient)
method2first_cluster_num_dict['partition_coefficient'] = est_k_partition_coefficient

est_k_LL, index_LL = last_leap_origin(all_centers, k_list)
print('For LL : Selected k =', est_k_LL)
method2first_cluster_num_dict['LL'] = est_k_LL

est_k_LML, index_LML = last_major_leap_origin(all_centers, k_list)
print('For LML : Selected k =', est_k_LML)

est_k_modified_partition_coefficient = k_list[index_modified_partition_coefficient.argmax()]
print('For modified_partition_coefficient : Selected k =', est_k_modified_partition_coefficient)
method2first_cluster_num_dict['modified_partition_coefficient'] = est_k_modified_partition_coefficient

est_k_partition_index = k_list[elbow_method(index_partition_index)]
print('For partition_index : Selected k =', est_k_partition_index)
method2first_cluster_num_dict['partition_index'] = est_k_partition_index

est_k_pbmf = k_list[index_pbmf.argmax()]
print('For pbmf : Selected k =', est_k_pbmf)
method2first_cluster_num_dict['pbmf'] = est_k_pbmf

est_k_pcaes = k_list[index_pcaes.argmax()]
print('For pcaes : Selected k =', est_k_pcaes)
method2first_cluster_num_dict['pcaes'] = est_k_pcaes

est_k_ren_liu_wang_yi = k_list[index_ren_liu_wang_yi.argmin()]
print('For ren_liu_wang_yi : Selected k =', est_k_ren_liu_wang_yi)
method2first_cluster_num_dict['ren_liu_wang_yi'] = est_k_ren_liu_wang_yi

index_rezaee = (sep_rezaee / sep_rezaee.max()) + (comp_rezaee / comp_rezaee.max())
est_k_rezaee = k_list[index_rezaee.argmin()]
print('For rezaee : Selected k =', est_k_rezaee)
method2first_cluster_num_dict['rezaee'] = est_k_rezaee

est_k_silhouette = k_list[index_silhouette.argmax()]
print('For silhouette : Selected k =', est_k_silhouette)
method2first_cluster_num_dict['silhouette'] = est_k_silhouette

index_slope_statistic = slope_statistic(index_silhouette, input_embed.shape[1])
est_k_slope_statistic = k_list[index_slope_statistic.argmax()]
print('For slope_statistic : Selected k =', est_k_slope_statistic)
method2first_cluster_num_dict['slope_statistic'] = est_k_slope_statistic

est_k_xie_beni = k_list[index_xie_beni.argmin()]
print('For xie_beni : Selected k =', est_k_xie_beni)
method2first_cluster_num_dict['xie_beni'] = est_k_xie_beni

est_k_xu_index = k_list[index_xu_index.argmin()]
print('For xu_index : Selected k =', est_k_xu_index)
method2first_cluster_num_dict['xu_index'] = est_k_xu_index

est_k_zhao_xu_franti = k_list[elbow_method(index_zhao_xu_franti)]
print('For zhao_xu_franti : Selected k =', est_k_zhao_xu_franti)
method2first_cluster_num_dict['zhao_xu_franti'] = est_k_zhao_xu_franti

print('Golden cluster number : ', cluster_num)
print()

if dataset == 'OPIEC' or dataset == 'reverb45k_change':
    print('Level two:')
    data = input_embed

    for method in method2first_cluster_num_dict:
        level_one_k = method2first_cluster_num_dict[method]
        print('Method:', method, 'level_one_k:', level_one_k)
        level_two_min, level_two_max, level_two_gap = level_one_k - level_one_gap, level_one_k + level_one_gap, int(
            level_one_gap / 10)
        minK, maxK = level_two_min, level_two_max
        cluster_list = range(level_two_min, level_two_max, level_two_gap)
        k_list = list(cluster_list)
        est_k = 0
        print('level_two_min, level_two_max, level_two_gap:', level_two_min, level_two_max, level_two_gap)

        if method == 'Inverse_JumpsMethod':
            jm = Inverse_JumpsMethod(data=input_embed, k_list=cluster_list, dim_is_bert=None)
            jm.Distortions(random_state=0)
            distortions = jm.distortions
            jm.Jumps(distortions=distortions)
            jumps = jm.jumps
            est_k = jm.recommended_cluster_number

        if method == 'JumpsMethod':
            jm = JumpsMethod(data=input_embed)
            jm.Distortions(random_state=0, cluster_list=cluster_list)
            distortions = jm.distortions
            jm.Jumps()
            est_k = jm.recommended_cluster_number

        if method == 'aic':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = aic(data, km1.cluster_centers_, km1.labels_)
            est_k = k_list[elbow_method(index)]

        if method == 'bic':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = bic(data, km1.cluster_centers_, km1.labels_)
            est_k = k_list[index.argmax()]

        if method == 'calinski_harabasz':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = calinski_harabasz(data, km1.cluster_centers_, km1.labels_)
            est_k = k_list[index.argmax()]

        if method == 'classification_entropy':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = classification_entropy(data, km1.cluster_centers_)
            est_k = k_list[index.argmin()]

        if method == 'compose_within_between':
            index = np.zeros((len(k_list)))
            km1 = KMeans(n_clusters=maxK, n_init=30, max_iter=300, tol=1e-9).fit(data)
            centerskmax = np.array(km1.cluster_centers_)
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    if k != k_max - 1:
                        km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                        index[i] = compose_within_between(data, km1.cluster_centers_, centerskmax)
            index[len(k_list) - 1] = compose_within_between(data, centerskmax, centerskmax)
            est_k = k_list[index.argmin()]

        if method == 'davies_bouldin':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = davies_bouldin(data, km1.cluster_centers_, km1.labels_)
            est_k = k_list[elbow_method(index)]

        if method == 'dunn':
            index = np.zeros((len(k_list)))
            pairwise_distances = cdist(data, data)
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = dunn(pairwise_distances, km1.labels_)
            est_k = k_list[index.argmax()]

        if method == 'elbow':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                index[i] = -km1.score(data)
            est_k = k_list[elbow_method(index)]

        if method == 'fukuyama_sugeno':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                index[i] = fukuyama_sugeno(data, km1.cluster_centers_)
            est_k = k_list[elbow_method(index)]

        if method == 'fuzzy_hypervolume':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                index[i] = fuzzy_hypervolume(data, km1.cluster_centers_)
            est_k = k_list[index.argmin()]

        if method == 'halkidi_vazirgannis':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = halkidi_vazirgannis(data, km1.cluster_centers_, km1.labels_)
            est_k = k_list[index.argmin()]

        if method == 'I_index':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = I_index(data, km1.cluster_centers_)
            est_k = k_list[index.argmax()]

        if method == 'partition_coefficient':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = partition_coefficient(data, km1.cluster_centers_)
            est_k = k_list[index.argmax()]

        if method == 'LL':
            all_centers = []
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    all_centers.append(km1.cluster_centers_)
            est_k, index = last_leap(all_centers, k_list)

        if method == 'modified_partition_coefficient':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = modified_partition_coefficient(data, km1.cluster_centers_)
            est_k = k_list[index.argmax()]

        if method == 'partition_index':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = partition_index(data, km1.cluster_centers_)
            est_k = k_list[elbow_method(index)]

        if method == 'pbmf':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = pbmf(data, km1.cluster_centers_)
            est_k = k_list[index.argmax()]

        if method == 'pcaes':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = pcaes(data, km1.cluster_centers_)
            est_k = k_list[index.argmax()]

        if method == 'ren_liu_wang_yi':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = ren_liu_wang_yi(data, km1.cluster_centers_, km1.labels_)
            est_k = k_list[index.argmin()]

        if method == 'rezaee':
            sep = np.zeros((len(k_list)))
            comp = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    sep[i], comp[i] = rezaee(data, km1.cluster_centers_)
            index = (sep / sep.max()) + (comp / comp.max())
            est_k = k_list[index.argmin()]

        if method == 'silhouette':
            pairwise_distances = cdist(data, data)
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = silhouette(pairwise_distances, km1.labels_)
            est_k = k_list[index.argmax()]

        if method == 'slope_statistic':
            pairwise_distances = cdist(data, data)
            sil = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    sil[i] = silhouette(pairwise_distances, km1.labels_)
            index = slope_statistic(sil, data.shape[1])
            est_k = k_list[index.argmax()]

        if method == 'xie_beni':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = xie_beni(data, km1.cluster_centers_)
            est_k = k_list[index.argmin()]

        if method == 'xu_index':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = xu_index(data, km1.cluster_centers_)
            est_k = k_list[index.argmin()]

        if method == 'zhao_xu_franti':
            index = np.zeros((len(k_list)))
            for i in range(len(k_list)):
                k = k_list[i]
                if k != 1:
                    km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                    index[i] = zhao_xu_franti(data, km1.cluster_centers_, km1.labels_)
            est_k = k_list[elbow_method(index)]

        print(method, ' k: ', est_k)
        print()

    print('Golden cluster number : ', cluster_num)
    print()