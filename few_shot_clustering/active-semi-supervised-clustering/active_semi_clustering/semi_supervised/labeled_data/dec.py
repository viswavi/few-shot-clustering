
from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans

from collections import namedtuple
import numpy as np
import random
import scipy.spatial.distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from tensorboardX import SummaryWriter
import torch
import torch.utils.data as util_data

from cmvc.Multi_view_CH_kmeans import init_seeded_kmeans_plusplus
from cmvc.test_performance import cluster_test

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../../sccl")

from sccl.models.Transformers import SCCLMatrix
from sccl.utils.optimizer import get_optimizer_linear_transformation
from sccl.training import MatrixDECTrainer


import time


class DEC(KMeans):
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

    def fit(self, X):

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
        Args = namedtuple("Args", "lr lr_scale eta temperature objective print_freq max_iter batch_size tensorboard")
        args = Args(1e-05, 100, 10, 0.5, "SCCL", 300, 2500 * X.shape[0] / self.batch_size, self.batch_size, tensorboard)


        optimizer = get_optimizer_linear_transformation(model, args, include_contrastive_loss=self.include_contrastive_loss, linear_transformation=self.linear_transformation)

        trainer = MatrixDECTrainer(model, optimizer, train_loader, X, self.labels, args, device=self.device, include_contrastive_loss=self.include_contrastive_loss, canonicalization_test_function=cluster_test, canonicalization_side_information=self.canonicalization_side_information)
        preds = trainer.train()
        self.labels_ = preds
        self.model_ = model

        # TODO(Vijay): do something with the output
        # TODO(Vijay): then, add appropriate noise and test with contrastive loss
