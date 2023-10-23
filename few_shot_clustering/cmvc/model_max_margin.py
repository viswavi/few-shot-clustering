#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from helper import *
import json
import logging
import os
import gensim
import numpy as np
from tqdm import tqdm

def hinge_loss(positive_score, negative_score, gamma):
    err = positive_score - negative_score + gamma
    max_err = err.clamp(0)
    return max_err


class KGEModel(nn.Module):
    def __init__(self, model_name, dict_local, init, E_init, R_init, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.embed_loc = dict_local
        self.E_init = E_init
        self.R_init = R_init
        self.init = init

        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        ''' Intialize embeddings '''
        if self.init == 'crawl':
            self.entity_embedding = nn.Parameter(torch.from_numpy(self.E_init))
            self.relation_embedding = nn.Parameter(torch.from_numpy(self.R_init))
        else:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        if model_name == 'pRotatE' or model_name == 'new_rotate':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE']:
            raise ValueError('model %s not supported' % model_name)

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = torch.norm(score, p=1, dim=2)
        return score

    
    @staticmethod
    def train_step(args, model, optimizer, train_iterator):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        negative_sample_size = int(args.single_negative_sample_size)
        gamma = torch.full((1, negative_sample_size), float(args.single_gamma))

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            gamma = gamma.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        positive_score = model(positive_sample)
        positive_score = positive_score.repeat(1, negative_sample_size)

        loss = hinge_loss(positive_score, negative_score, gamma)

        if args.uni_weight:
            loss = loss.sum()
        else:
            loss = (subsampling_weight * loss).sum()/subsampling_weight.sum()

        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(p=3)**3 +
                model.relation_embedding.norm(p=3).norm(p=3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return log

    @staticmethod
    def cross_train_step(args, model, optimizer, seed_iterator):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        negative_sample_size = int(args.cross_negative_sample_size)
        gamma = torch.full((1, negative_sample_size), float(args.cross_gamma))  # 返回大小为sizes,单位值为fill_value的矩阵

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, seed_sim, mode = next(seed_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            gamma = gamma.cuda()
            seed_sim = seed_sim.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        positive_score = model(positive_sample)
        seed_sim = torch.from_numpy(np.diag(seed_sim.t().cpu().numpy()[0])).cuda()

        positive_score = positive_score.repeat(1, negative_sample_size)

        loss = hinge_loss(positive_score, negative_score, gamma)
        loss = loss.sum(dim=1) * seed_sim

        if args.uni_weight:
            loss = loss.sum()
        else:
            loss = (subsampling_weight * loss).sum() / subsampling_weight.sum()


        if args.regularization != 0.0:
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'loss': loss.item()
        }

        return log

    def get_seeds(self, p, side_info, logging):
        self.p = p
        self.side_info = side_info
        self.logging = logging
        self.id2ent, self.id2rel = self.side_info.id2ent, self.side_info.id2rel
        self.ent2id, self.rel2id = self.side_info.ent2id, self.side_info.rel2id
        self.ent2triple_id_list, self.rel2triple_id_list = self.side_info.ent2triple_id_list, self.side_info.rel2triple_id_list
        self.trpIds = self.side_info.trpIds
        entity_embedding, relation_embedding = self.entity_embedding.data, self.relation_embedding.data
        self.seed_trpIds, self.seed_sim = [], []
        for i in tqdm(range(len(entity_embedding))):
            for j in range(i + 1, len(entity_embedding)):
                e1_embed, e2_embed = entity_embedding[i], entity_embedding[j]
                sim = torch.cosine_similarity(e1_embed, e2_embed, dim=0)
                if sim > self.p.entity_threshold:
                    ent1, ent2 = self.id2ent[i], self.id2ent[j]
                    for ent in [ent1, ent2]:
                        triple_list = self.ent2triple_id_list[ent]
                        for triple_id in triple_list:
                            triple = self.trpIds[triple_id]
                            if str(self.id2ent[triple[0]]) == str(ent1):
                                trp = (self.ent2id[str(ent2)], triple[1], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2ent[triple[0]]) == str(ent2):
                                trp = (self.ent2id[str(ent1)], triple[1], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2ent[triple[2]]) == str(ent1):
                                trp = (triple[0], triple[1], self.ent2id[str(ent2)])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2ent[triple[2]]) == str(ent2):
                                trp = (triple[0], triple[1], self.ent2id[str(ent1)])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)

        for i in tqdm(range(len(relation_embedding))):
            for j in range(i + 1, len(relation_embedding)):
                r1_embed, r2_embed = relation_embedding[i], relation_embedding[j]
                sim = torch.cosine_similarity(r1_embed, r2_embed, dim=0)
                if sim > self.p.relation_threshold:
                    rel1, rel2 = self.id2rel[i], self.id2rel[j]
                    for rel in [rel1, rel2]:
                        triple_list = self.rel2triple_id_list[rel]
                        for triple_id in triple_list:
                            triple = self.trpIds[triple_id]
                            if str(self.id2rel[triple[1]]) == str(rel1):
                                trp = (triple[0], self.rel2id[str(rel2)], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2rel[triple[1]]) == str(rel2):
                                trp = (triple[0], self.rel2id[str(rel1)], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
        return self.seed_trpIds, self.seed_sim

    def set_logger(self):
        '''
        Write logs to checkpoint and console
        '''

        if self.p.do_train:
            log_file = os.path.join(self.p.out_path or self.p.init_checkpoint, 'train.log')
        else:
            log_file = os.path.join(self.p.out_path or self.p.init_checkpoint, 'test.log')

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def log_metrics(self, mode, step, metrics):
        '''
        Print the evaluation logs
        '''
        for metric in metrics:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))