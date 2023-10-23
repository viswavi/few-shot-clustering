
from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans

from collections import namedtuple
import json
import numpy as np
import random
import scipy.spatial.distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from tensorboardX import SummaryWriter
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import torch
import torch.utils.data as util_data

from cmvc.Multi_view_CH_kmeans import init_seeded_kmeans_plusplus
from cmvc.test_performance import cluster_test

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../../sccl")

from sccl.models.Transformers import SCCLMatrix, SCCLBertTransE
from sccl.utils.optimizer import get_optimizer_linear_transformation, get_optimizer_deep_sccl
from sccl.training import MatrixSCCLTrainer, DeepSCCLTrainer

class SCCL(KMeans):
    def __init__(self,
                 n_clusters=3,
                 max_iter=100,
                 normalize_vectors=False,
                 split_normalization=False,
                 verbose=False,
                 cluster_init="random",
                 device="cuda:0",
                 include_contrastive_loss=False,
                 labels=None,
                 batch_size = 400,
                 linear_transformation = False,
                 canonicalization_side_information=None,
                 tensorboard_parent_dir="/projects/ogma1/vijayv/okb-canonicalization/clustering/sccl/",
                 tensorboard_dir="tmp"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.normalize_vectors = normalize_vectors
        self.split_normalization = split_normalization
        self.verbose = verbose 
        self.batch_size = batch_size
        self.num_dataloader_workers = 4
        self.init = cluster_init
        self.device=device
        self.include_contrastive_loss=include_contrastive_loss
        assert labels is not None
        self.labels = labels
        self.linear_transformation = linear_transformation
        self.canonicalization_side_information = canonicalization_side_information
        self.tensorboard_parent_dir = tensorboard_parent_dir
        self.tensorboard_dir = tensorboard_dir

    def fit(self, X, pairwise_constraints):

        if self.init == "k-means":
            clusterer = KMeans(n_clusters=self.n_clusters, normalize_vectors=self.normalize_vectors, split_normalization=self.split_normalization, init="k-means++", num_reinit=1, verbose=False)
            clusterer.fit(X)
            labels = clusterer.labels_
            clusters_list = {}
            for l, feat in zip(labels, X):
                if l not in clusters_list:
                    clusters_list[l] = []
                clusters_list[l].append(feat)
            
            cluster_centers = np.empty((self.n_clusters, X.shape[1]))
            for i, l in enumerate(clusters_list.keys()):
                avg_vec = np.mean(clusters_list[l], axis=0)
                cluster_centers[i] = avg_vec
        else:
            cluster_centers = self._init_cluster_centers(X)

        torch.cuda.set_device(self.device)

        train_loader = util_data.DataLoader(X.astype("float32"), batch_size=self.batch_size, shuffle=True, num_workers=1)

        model = SCCLMatrix(emb_size=X.shape[1], cluster_centers=cluster_centers, include_contrastive_loss=self.include_contrastive_loss, linear_transformation = self.linear_transformation) 
        model = model.cuda()

        # tensorboard_dir could be "opiec_59k_sccl_with_unsupervised_cl_no_linear_transformation/"
        resDir = os.path.join(self.tensorboard_parent_dir, self.tensorboard_dir)
        resPath = "SCCL.tensorboard"
        resPath = resDir + resPath
        tensorboard = SummaryWriter(resPath)

        # optimize  
        Args = namedtuple("Args", "lr lr_scale lr_scale_scl eta temperature objective print_freq max_iter batch_size tensorboard")
        args = Args(5e-06, 100, 100, 10, 0.5, "SCCL", 300, 2000 * X.shape[0] / self.batch_size, self.batch_size, tensorboard)


        optimizer = get_optimizer_linear_transformation(model, args, include_contrastive_loss=self.include_contrastive_loss, linear_transformation=self.linear_transformation)

        trainer = MatrixSCCLTrainer(model, optimizer, train_loader, X, pairwise_constraints, self.labels, args, device=self.device, include_contrastive_loss=self.include_contrastive_loss, canonicalization_test_function=cluster_test, canonicalization_side_information=self.canonicalization_side_information)
        preds = trainer.train()
        self.labels_ = preds
        self.model_ = model

class DeepSCCL(KMeans):
    def __init__(self,
                 bert_model,
                 kge_model,
                 n_clusters=3,
                 max_iter=100,
                 normalize_vectors=False,
                 split_normalization=False,
                 verbose=False,
                 cluster_init="random",
                 device="cuda:1",
                 include_contrastive_loss=False,
                 labels=None,
                 batch_size = 6,
                 linear_transformation = False,
                 canonicalization_side_information=None,
                 tensorboard_parent_dir="/projects/ogma1/vijayv/okb-canonicalization/clustering/deep_sccl/",
                 tensorboard_dir="tmp"):
        self.bert = bert_model
        self.kge_model = kge_model
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.normalize_vectors = normalize_vectors
        self.split_normalization = split_normalization
        self.verbose = verbose 
        self.batch_size = batch_size
        self.num_dataloader_workers = 4
        self.init = cluster_init
        self.device=device
        self.include_contrastive_loss=include_contrastive_loss
        assert labels is not None
        self.labels = labels
        self.linear_transformation = linear_transformation
        self.canonicalization_side_information = canonicalization_side_information
        self.tensorboard_parent_dir = tensorboard_parent_dir
        self.tensorboard_dir = tensorboard_dir

    def fit(self, X, pairwise_constraints):

        if self.init == "k-means":
            clusterer = KMeans(n_clusters=self.n_clusters, normalize_vectors=self.normalize_vectors, split_normalization=self.split_normalization, init="k-means++", num_reinit=1, verbose=False)
            clusterer.fit(X)
            labels = clusterer.labels_
            clusters_list = {}
            for l, feat in zip(labels, X):
                if l not in clusters_list:
                    clusters_list[l] = []
                clusters_list[l].append(feat)
            
            cluster_centers = np.empty((self.n_clusters, X.shape[1]))
            for i, l in enumerate(clusters_list.keys()):
                avg_vec = np.mean(clusters_list[l], axis=0)
                cluster_centers[i] = avg_vec
        else:
            cluster_centers = self._init_cluster_centers(X)

        torch.cuda.set_device(self.device)

        train_loader = util_data.DataLoader(X.astype("float32"), batch_size=self.batch_size, shuffle=True, num_workers=1)

        side_info = self.canonicalization_side_information.side_info
        
        ent_ids = []
        ents = []
        sentences = []
        noised_ent_ids = []
        noised_sentences_cache = os.path.join("/projects/ogma1/vijayv/okb-canonicalization/clustering/file/OPIEC59k_test", "noised_sentences.json")
        if os.path.exists(noised_sentences_cache):
            noised_sentences = json.load(open(noised_sentences_cache, 'r'))
            run_sentence_noising = False
        else:
            noised_sentences = []
            run_sentence_noising = True

        if run_sentence_noising:
            augmenter1 = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", aug_min=1, aug_p=0.2, device="cuda:2")

        sentence_unprocessing_mapping = json.load(open("/projects/ogma1/vijayv/okb-canonicalization/clustering/file/gpt3_cache/sentence_unprocessing_map.json"))

        for i in tqdm(range(len(X))):
            one_hot_vector = np.zeros(X.shape[0], dtype=np.float32)
            one_hot_vector[i] = 1
            entity_sentence_idxs = side_info.ent_id2sentence_list[i]
            all_sentences = [side_info.sentence_List[j] for j in entity_sentence_idxs]
            _ = '''
            Choose longest sentence under 306 characers, as in
            https://github.com/Yang233666/cmvc/blob/6e752b1aa5db7ff99eb2fa73476e392a00b0b89a/Context_view.py#L98
            '''
            longest_sentences = sorted([s for s in all_sentences if len(s) < 306], key=len)
            try:
                if len(longest_sentences) == 0:
                    context_sentence = sentence_unprocessing_mapping[all_sentences[0]]
                else:
                    context_sentence = sentence_unprocessing_mapping[longest_sentences[0]]
            except:
                breakpoint()
            ents.append(side_info.id2ent[i])
            ent_ids.append(one_hot_vector)
            one_hot_noise = np.random.normal(loc=0.0, scale=0.05, size=one_hot_vector.size).astype(np.float32)
            noised_ent_ids.append(one_hot_vector + one_hot_noise)
            if len(longest_sentences) == 0:
                sentences.append(context_sentence)
            else:
                sentences.append(context_sentence)
            if run_sentence_noising:
                noised_sentence = augmenter1.augment(context_sentence)
                noised_sentences.append(noised_sentence[0])

        if run_sentence_noising:
            json.dump(noised_sentences, open(noised_sentences_cache, 'w'), indent=4)

        entities_and_sentences = list(zip(ent_ids, noised_ent_ids, sentences, noised_sentences, self.labels))
        train_loader = util_data.DataLoader(entities_and_sentences, batch_size=self.batch_size, shuffle=True, num_workers=1)
        test_loader = util_data.DataLoader(entities_and_sentences, batch_size=self.batch_size, shuffle=False, num_workers=1)


        model = SCCLBertTransE(X, self.bert, self.kge_model, emb_size=X.shape[1], cluster_centers=cluster_centers, include_contrastive_loss=self.include_contrastive_loss, linear_transformation = self.linear_transformation, canonicalization_side_information=self.canonicalization_side_information) 
        model = model.cuda()

        # tensorboard_dir could be "opiec_59k_sccl_with_unsupervised_cl_no_linear_transformation/"
        resDir = os.path.join(self.tensorboard_parent_dir, self.tensorboard_dir)
        resPath = "SCCL.tensorboard"
        resPath = resDir + resPath
        tensorboard = SummaryWriter(resPath)

        # optimize
        Args = namedtuple("Args", "lr lr_scale lr_scale_scl eta temperature objective print_freq max_iter batch_size tensorboard")
        args = Args(5e-06, 100, 100, 10, 0.5, "SCCL", 300, 2000 * X.shape[0] / self.batch_size, self.batch_size, tensorboard)


        optimizer = get_optimizer_deep_sccl(model, args, include_contrastive_loss=self.include_contrastive_loss, linear_transformation=self.linear_transformation)

        trainer = DeepSCCLTrainer(model, optimizer, train_loader, test_loader, entities_and_sentences, X, pairwise_constraints, self.labels, args, device=self.device, include_contrastive_loss=self.include_contrastive_loss, canonicalization_test_function=cluster_test, canonicalization_side_information=self.canonicalization_side_information)
        preds = trainer.train()
        self.labels_ = preds
        self.model_ = model
