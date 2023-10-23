import gensim, itertools, pickle, random, time
from helper import *
from cmvc_utils import cos_sim
from test_performance import cluster_test, HAC_getClusters
from train_embedding_model import Train_Embedding_Model, pair2triples
from Context_view import BERT_Model

class DisjointSet(object):
    def __init__(self):
        self.leader = {}  # maps a member to the group's leader
        self.group = {}  # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return  # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])


def amieInfo(triples, ent2id, rel2id):
    uf = DisjointSet()
    min_supp = 2
    min_conf = 0.5  # cesi=0.2
    amie_cluster = []
    rel_so = {}

    for trp in triples:
        sub, rel, obj = trp['triple']
        if sub in ent2id and rel in rel2id and obj in ent2id:
            sub_id, rel_id, obj_id = ent2id[sub], rel2id[rel], ent2id[obj]
            rel_so[rel_id] = rel_so.get(rel_id, set())
            rel_so[rel_id].add((sub_id, obj_id))

    for r1, r2 in itertools.combinations(rel_so.keys(), 2):
        supp = len(rel_so[r1].intersection(rel_so[r2]))
        if supp < min_supp: continue

        s1, _ = zip(*list(rel_so[r1]))
        s2, _ = zip(*list(rel_so[r2]))

        z_conf_12, z_conf_21 = 0, 0
        for ele in s1:
            if ele in s2: z_conf_12 += 1
        for ele in s2:
            if ele in s1: z_conf_21 += 1

        conf_12 = supp / z_conf_12
        conf_21 = supp / z_conf_21

        if conf_12 >= min_conf and conf_21 >= min_conf:
            amie_cluster.append((r1, r2))  # Replace with union find DS
            uf.add(r1, r2)

    rel2amie = uf.leader
    return rel2amie


def seed_pair2cluster(seed_pair_list, ent_list):
    pair_dict = dict()
    for seed_pair in seed_pair_list:
        a, b = seed_pair
        if a != b:
            if a < b:
                rep, ent_id = a, b
            else:
                ent_id, rep = b, a
            if ent_id not in pair_dict:
                if rep not in pair_dict:
                    pair_dict.update({ent_id: rep})
                else:
                    new_rep = pair_dict[rep]
                    j = 0
                    while rep in pair_dict:
                        new_rep = pair_dict[rep]
                        rep = new_rep
                        j += 1
                        if j > 1000000:
                            break
                    pair_dict.update({ent_id: new_rep})
            else:
                if rep not in pair_dict:
                    new_rep = pair_dict[ent_id]
                    if rep > new_rep:
                        pair_dict.update({rep: new_rep})
                    else:
                        pair_dict.update({new_rep: rep})
                else:
                    old_rep = rep
                    new_rep = pair_dict[rep]
                    j = 0
                    while rep in pair_dict:
                        new_rep = pair_dict[rep]
                        rep = new_rep
                        j += 1
                        if j > 1000000:
                            break
                    if old_rep > new_rep:
                        pair_dict.update({ent_id: new_rep})
                    else:
                        pair_dict.update({ent_id: old_rep})

    cluster_list = []
    for i in range(len(ent_list)):
        cluster_list.append(i)
    for ent_id in pair_dict:
        rep = pair_dict[ent_id]
        if ent_id < len(cluster_list):
            cluster_list[ent_id] = rep
    return cluster_list


def get_seed_pair(ent_list, ent2id, ent_old_id2new_id):
    seed_pair = []
    for i in range(len(ent_list)):
        ent1 = ent_list[i]
        old_id1 = ent2id[ent1]
        if old_id1 in ent_old_id2new_id:
            for j in range(i + 1, len(ent_list)):
                ent2 = ent_list[j]
                old_id2 = ent2id[ent2]
                if old_id2 in ent_old_id2new_id:
                    new_id1, new_id2 = ent_old_id2new_id[old_id1], ent_old_id2new_id[
                        old_id2]
                    if new_id1 == new_id2:
                        id_tuple = (i, j)
                        seed_pair.append(id_tuple)
    return seed_pair


def difference_cluster2pair(cluster_list_1, cluster_list_2, EL_seed):
    new_seed_pair_list = []
    for i in range(len(cluster_list_1)):
        id_1, id_2 = cluster_list_1[i], cluster_list_2[i]
        if id_1 == id_2:
            continue
        else:
            index_list_1 = [i for i, x in enumerate(cluster_list_1) if x == id_1]
            index_list_2 = [i for i, x in enumerate(cluster_list_2) if x == id_2]
            if len(index_list_2) == 1:
                continue
            else:
                iter_list_1 = list(itertools.combinations(index_list_1, 2))
                iter_list_2 = list(itertools.combinations(index_list_2, 2))
                if len(iter_list_1) > 0:
                    for iter_pair in iter_list_1:
                        if iter_pair in iter_list_2: iter_list_2.remove(iter_pair)
                for iter in iter_list_2:
                    if iter not in EL_seed:
                        new_seed_pair_list.append(iter)
    return new_seed_pair_list


def totol_cluster2pair(cluster_list):
    seed_pair_list, id_list = [], []
    for i in range(len(cluster_list)):
        id = cluster_list[i]
        if id not in id_list:
            id_list.append(id)
            index_list = [i for i, x in enumerate(cluster_list) if x == id]
            if len(index_list) > 1:
                iter_list = list(itertools.combinations(index_list, 2))
                seed_pair_list += iter_list
    return seed_pair_list

def initialize_cluster_seeds(num_cluster_seeds, ent2id, id_to_embedding_rows, true_clust2ent):
    """Init k-means clusters seeds according to k-means++

    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    num_cluster_seeds : array or sparse matrix, shape (n_samples, n_features)
        Pre-initialized cluster seeds chosen by a previous method (such as an
        oracle). The number of initial cluster seeds N must be less than
        num_clusters.
    ...

    Return indices into the embedding matrix to use as cluster initializations
    """
    seed_points = []
    assert num_cluster_seeds <= len(true_clust2ent)
    cluster_names_to_sample = random.sample(true_clust2ent.keys(), num_cluster_seeds)
    for cluster_name in cluster_names_to_sample:
        cluster_entities = true_clust2ent[cluster_name]
        random_entity = random.choice(list(cluster_entities))
        entity_name = random_entity.split("|")[0]
        seed_points.append(id_to_embedding_rows[ent2id[entity_name]])
    return seed_points



class Embeddings(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, true_ent2clust, true_clust2ent, sub_uni2triple_dict=None,
                 triple_list=None, num_reinit=10, random_seed=None):
        self.p = params

        self.side_info = side_info
        self.ent2embed = {}  # Stores final embeddings learned for noun phrases
        self.rel2embed = {}  # Stores final embeddings learned for relation phrases
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.sub_uni2triple_dict = sub_uni2triple_dict
        self.triples_list = triple_list

        self.rel_id2sentence_list = dict()
        self.num_reinit = num_reinit
        self.random_seed = random_seed

        ent_id2sentence_list = self.side_info.ent_id2sentence_list
        for rel in self.side_info.rel_list:
            rel_id = self.side_info.rel2id[rel]
            if rel_id not in self.rel_id2sentence_list:
                triple_id_list = self.side_info.rel2triple_id_list[rel]
                sentence_list = []
                for triple_id in triple_id_list:
                    triple = self.triples_list[triple_id]
                    sub, rel_, obj = triple['triple'][0], triple['triple'][1], triple['triple'][2]
                    assert str(rel_) == str(rel)
                    if sub in self.side_info.ent2id:
                        sentence_list += ent_id2sentence_list[self.side_info.ent2id[sub]]
                    if obj in self.side_info.ent2id:
                        sentence_list += ent_id2sentence_list[self.side_info.ent2id[obj]]
                sentence_list = list(set(sentence_list))
                self.rel_id2sentence_list[rel_id] = sentence_list
        print('self.rel_id2sentence_list:', type(self.rel_id2sentence_list), len(self.rel_id2sentence_list))

    def fit(self):

        show_memory = False
        if show_memory:
            print('show_memory:', show_memory)
            import tracemalloc
            tracemalloc.start(25)  # 默认25个片段，这个本质还是多次采样

        clean_ent_list, clean_rel_list = [], []
        for ent in self.side_info.ent_list: clean_ent_list.append(ent.split('|')[0])
        for rel in self.side_info.rel_list: clean_rel_list.append(rel.split('|')[0])

        print('clean_ent_list:', type(clean_ent_list), len(clean_ent_list))
        print('clean_rel_list:', type(clean_rel_list), len(clean_rel_list))

        ''' Intialize embeddings '''
        if self.p.embed_init == 'crawl':
            fname1, fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/1E_init', '../file/' + self.p.dataset + '_' + self.p.split + '/1R_init'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate pre-trained embeddings')

                model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
                self.E_init = getEmbeddings(model, clean_ent_list, self.p.embed_dims)
                self.R_init = getEmbeddings(model, clean_rel_list, self.p.embed_dims)

                pickle.dump(self.E_init, open(fname1, 'wb'))
                pickle.dump(self.R_init, open(fname2, 'wb'))
            else:
                print('load init embeddings')
                self.E_init = pickle.load(open(fname1, 'rb'))
                self.R_init = pickle.load(open(fname2, 'rb'))

        else:
            print('generate init random embeddings')
            self.E_init = np.random.rand(len(clean_ent_list), self.p.embed_dims)
            self.R_init = np.random.rand(len(clean_rel_list), self.p.embed_dims)

        folder = 'multi_view/relation_view'
        print('folder:', folder)
        folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/'
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)

        fname_EL = '../file/' + self.p.dataset + '_' + self.p.split + '/EL_seed'
        if not checkFile(fname_EL):
            self.EL_seed = get_seed_pair(self.side_info.ent_list, self.side_info.ent2id,
                                            self.side_info.ent_old_id2new_id)
            pickle.dump(self.EL_seed, open(fname_EL, 'wb'))
        else:
            self.EL_seed = pickle.load(open(fname_EL, 'rb'))
        print('self.EL_seed:', type(self.EL_seed), len(self.EL_seed))

        fname_amie = '../file/' + self.p.dataset + '_' + self.p.split + '/amie_rp_seed'
        if not checkFile(fname_amie):
            self.amie_rp = amieInfo(self.triples_list, self.side_info.ent2id, self.side_info.rel2id)
            self.amie_rp_seed = get_seed_pair(self.side_info.rel_list, self.side_info.rel2id,
                                            self.amie_rp)
            pickle.dump(self.amie_rp_seed, open(fname_amie, 'wb'))
        else:
            self.amie_rp_seed = pickle.load(open(fname_amie, 'rb'))
        print('self.amie_rp_seed:', type(self.amie_rp_seed), len(self.amie_rp_seed))

        web_seed_Jaccard_threshold = 0.015
        fname2_entity = '../file/' + self.p.dataset + '_' + self.p.split + '/WEB_seed/entity/cluster_list_threshold_' + \
                        str(web_seed_Jaccard_threshold) + '_url_max_length_all'
        fname2_relation = '../file/' + self.p.dataset + '_' + self.p.split + '/WEB_seed/relation/cluster_list_threshold_' + \
                          str(web_seed_Jaccard_threshold) + '_url_max_length_all'
        print('fname2_entity:', fname2_entity)
        print('fname2_relation:', fname2_relation)
        self.web_entity_cluster_list = pickle.load(open(fname2_entity, 'rb'))
        self.web_relation_cluster_list = pickle.load(open(fname2_relation, 'rb'))
        print('self.web_entity_cluster_list:', type(self.web_entity_cluster_list),
              len(self.web_entity_cluster_list),
              self.web_entity_cluster_list[0:10])
        print('self.web_relation_cluster_list:', type(self.web_relation_cluster_list),
              len(self.web_relation_cluster_list),
              self.web_relation_cluster_list[0:10])

        self.web_entity_seed_pair_list = totol_cluster2pair(self.web_entity_cluster_list)
        self.web_relation_seed_pair_list = totol_cluster2pair(self.web_relation_cluster_list)
        print('self.web_entity_seed_pair_list:', type(self.web_entity_seed_pair_list),
              len(self.web_entity_seed_pair_list), self.web_entity_seed_pair_list[0:10])
        print('self.web_relation_seed_pair_list:', type(self.web_relation_seed_pair_list),
              len(self.web_relation_seed_pair_list), self.web_relation_seed_pair_list[0:10])

        # web_entity_cluster_list = seed_pair2cluster(self.web_entity_seed_pair_list, clean_ent_list)
        # cluster_test(self.p, self.side_info, web_entity_cluster_list, self.true_ent2clust, self.true_clust2ent,
        #              print_or_not=True)

        # web_relation_cluster_list = seed_pair2cluster(self.web_relation_seed_pair_list, clean_rel_list)
        # print('web_relation_cluster_list:', type(web_relation_cluster_list), len(web_relation_cluster_list),
        #       web_relation_cluster_list[0:10])
        # print('different web_relation_cluster:', len(list(set(web_relation_cluster_list))))
        # print()

        # el_cluster_list = seed_pair2cluster(self.EL_seed, clean_ent_list)
        # print('el_cluster_list:', type(el_cluster_list), len(el_cluster_list), el_cluster_list[0:10])
        # cluster_test(self.p, self.side_info, el_cluster_list, self.true_ent2clust, self.true_clust2ent,
        #              print_or_not=True)

        self.all_seed_pair_list = []
        for pair in self.web_entity_seed_pair_list:
            if pair not in self.all_seed_pair_list:
                self.all_seed_pair_list.append(pair)
        for pair in self.EL_seed:
            if pair not in self.all_seed_pair_list:
                self.all_seed_pair_list.append(pair)
        self.context_seed_pair_list = []
        for pair in self.web_relation_seed_pair_list:
            if pair not in self.context_seed_pair_list:
                self.context_seed_pair_list.append(pair)
        for pair in self.amie_rp_seed:
            if pair not in self.context_seed_pair_list:
                self.context_seed_pair_list.append(pair)
        all_cluster_list = seed_pair2cluster(self.all_seed_pair_list, clean_ent_list)
        # cluster_test(self.p, self.side_info, all_cluster_list, self.true_ent2clust, self.true_clust2ent,
        #              print_or_not=True)
        print('self.context_seed_pair_list:', type(self.context_seed_pair_list), len(self.context_seed_pair_list),
              self.context_seed_pair_list[0:10])
        context_relation_cluster_list = seed_pair2cluster(self.context_seed_pair_list, clean_rel_list)
        print('context_relation_cluster_list:', type(context_relation_cluster_list), len(context_relation_cluster_list),
              context_relation_cluster_list[0:10])
        print('different context_relation_cluster_list:', len(list(set(context_relation_cluster_list))))
        print()

        relation_seed_pair_list = self.all_seed_pair_list
        relation_seed_cluster_list = seed_pair2cluster(relation_seed_pair_list, clean_ent_list)
        print('fact view seed :')
        cluster_test(self.p, self.side_info, relation_seed_cluster_list, self.true_ent2clust, self.true_clust2ent,
                     print_or_not=True)
        self.seed_trpIds, self.seed_sim = pair2triples(relation_seed_pair_list, clean_ent_list, self.side_info.ent2id,
                                                       self.side_info.id2ent, self.side_info.ent2triple_id_list,
                                                       self.side_info.trpIds, self.E_init, cos_sim, is_cuda=False,
                                                       high_confidence=False)
        print('self.seed_trpIds:', type(self.seed_trpIds), len(self.seed_trpIds), self.seed_trpIds[0:30])
        print('self.seed_sim:', type(self.seed_sim), len(self.seed_sim), self.seed_sim[0:30])
        print()

        if self.p.use_Embedding_model:
            fname1, fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/entity_embedding', '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/relation_embedding'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate TransE embeddings', fname1)
                self.new_seed_trpIds, self.new_seed_sim = self.seed_trpIds, self.seed_sim
                entity_embedding, relation_embedding = self.E_init, self.R_init
                print('self.training_time', 'use pre-trained crawl embeddings ... ')

                TEM = Train_Embedding_Model(self.p, self.side_info, entity_embedding, relation_embedding,
                                            relation_seed_pair_list, self.new_seed_trpIds, self.new_seed_sim)
                self.entity_embedding, self.relation_embedding = TEM.train()

                pickle.dump(self.entity_embedding, open(fname1, 'wb'))
                pickle.dump(self.relation_embedding, open(fname2, 'wb'))
            else:
                print('load TransE embeddings')
                self.entity_embedding = pickle.load(open(fname1, 'rb'))
                self.relation_embedding = pickle.load(open(fname2, 'rb'))

            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.entity_embedding[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.relation_embedding[id]

        else:  # do not use embedding model
            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.E_init[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.R_init[id]

        if self.p.input == 'entity':
            context_view_label = all_cluster_list
            print('context_view_seed : web_entity + EL')
            cluster_test(self.p, self.side_info, context_view_label, self.true_ent2clust, self.true_clust2ent, print_or_not=True)
        else:
            context_view_label = context_relation_cluster_list
            print('context_view_seed : web_relation + AMIE')

        folder = 'multi_view/context_view_' + str(self.p.input)
        print('self.p.input:', self.p.input)
        print('folder:', folder)
        print()
        self.epochs = 3
        # self.epochs = 1  # this is the rebuttal mode
        print('self.epochs:', self.epochs)

        if self.p.use_context and self.p.use_BERT:
            folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/'
            if not os.path.exists(folder_to_make):
                os.makedirs(folder_to_make)
            fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/' + 'BERT_fine-tune_label'
            print('fname1:', fname1)
            if not checkFile(fname1):
                print('generate BERT_fine-tune_', fname1)
                self.label = context_view_label
                self.sub_label = []
                for eid in self.side_info.isSub.keys():
                    self.sub_label.append(self.label[eid])
                K_init = len(list(set(self.sub_label)))
                print('K_init:', K_init)
                print('epochs:', self.epochs)
                for random_seed in range(self.epochs):
                    BERT_self_training_time = random_seed
                    if str(self.p.input) == 'entity':
                        input_list = clean_ent_list
                    else:
                        input_list = self.side_info.rel_list
                    K = K_init  # just for cesi_main_opiec_2
                    print('K:', K)
                    BM = BERT_Model(self.p, self.side_info, input_list, self.label,
                                    self.true_ent2clust, self.true_clust2ent, 0,
                                    BERT_self_training_time, self.sub_uni2triple_dict, self.rel_id2sentence_list, K)
                    self.label, K_init = BM.fine_tune()
                pickle.dump(self.label, open(fname1, 'wb'))
            else:
                print('load BERT_fine-tune_', fname1)
                self.label = pickle.load(open(fname1, 'srb'))
                context_view_label = self.label
        old_label, new_label = context_view_label, self.label
        print('old_label : ')
        cluster_test(self.p, self.side_info, old_label, self.true_ent2clust, self.true_clust2ent, print_or_not=True)
        print('new_label : ')
        cluster_test(self.p, self.side_info, new_label, self.true_ent2clust, self.true_clust2ent, print_or_not=True)

        BERT_self_training_time = self.epochs - 1
        fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/bert_cls_el_' + str(0) + '_' + str(
            BERT_self_training_time)
        # fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/bert_cls_el_' + str(0) + '_' + str(
        #     BERT_self_training_time) + '_finetune'  # this is the rebuttal mode
        self.BERT_CLS = pickle.load(open(fname1, 'rb'))

        print('self.ent2embed:', len(self.ent2embed))
        print('self.rel2embed:', len(self.rel2embed))
        print('self.BERT_CLS:', len(self.BERT_CLS))

        self.relation_view_embed, self.context_view_embed = [], []
        id_to_embedding_rows = {}
        for ent in clean_ent_list:
            id = self.side_info.ent2id[ent]
            if id in self.side_info.isSub:
                self.relation_view_embed.append(self.ent2embed[id])
                id_to_embedding_rows[id] = len(self.context_view_embed)
                self.context_view_embed.append(self.BERT_CLS[id])
        print('self.relation_view_embed:', len(self.relation_view_embed))
        print('self.context_view_embed:', len(self.context_view_embed))

        print('fact view: ')
        # threshold_or_cluster = 'threshold'
        threshold_or_cluster = 'cluster'
        if threshold_or_cluster == 'threshold':
            if self.p.dataset == 'OPIEC59k':
                cluster_threshold_real = 0.29
            else:
                cluster_threshold_real = 0.35
        else:
            if self.p.dataset == 'OPIEC59k':
                cluster_threshold_real = 490
            else:
                cluster_threshold_real = 6700
        print('cluster_threshold_real:', cluster_threshold_real)
        labels, clusters_center = HAC_getClusters(self.p, self.relation_view_embed, cluster_threshold_real, False)
        cluster_predict_list = list(labels)
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
            = cluster_test(self.p, self.side_info, cluster_predict_list, self.true_ent2clust,
                           self.true_clust2ent)
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()
        # exit()

        print('context view: ')
        if self.p.dataset == 'OPIEC59k':
            cluster_threshold_real = 923
        else:
            cluster_threshold_real = 7064
        print('cluster_threshold_real:', cluster_threshold_real)
        labels, clusters_center = HAC_getClusters(self.p, self.context_view_embed, cluster_threshold_real, True)
        cluster_predict_list = list(labels)
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
            = cluster_test(self.p, self.side_info, cluster_predict_list, self.true_ent2clust,
                           self.true_clust2ent)
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()

        print('Model is multi-view spherical-k-means')
        np.save(open("../data/OPIEC59k/valid_relation_view_embed.npz", 'wb'), np.vstack(self.relation_view_embed))
        np.save(open("../data/OPIEC59k/valid_context_view_embed.npz", 'wb'), np.vstack(self.context_view_embed))
        json.dump(self.side_info.id2sub, open("../data/OPIEC59k/valid_entId2name.json", 'w'), indent=4)



        for random_seed in range(3):
            print('test time:', random_seed)
            if self.p.dataset == 'OPIEC59k':
                n_cluster = 490
            elif self.p.dataset == 'reverb45k' or self.p.dataset == 'reverb45k_change':
                n_cluster = 6700
            else:
                n_cluster = 5700

            print('n_cluster:', type(n_cluster), n_cluster)

            t0 = time.time()
            real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
            print('time:', real_time)

            if self.p.kmeans_initialization == "seeded-k-means++":
                init = "seeded-k-means++"
                assert self.p.num_cluster_seeds <= n_cluster
                seed_set = initialize_cluster_seeds(self.p.num_cluster_seeds, self.side_info.ent2id, id_to_embedding_rows, self.true_clust2ent)
                print("TODO(Vijay): Not implemented")
            elif self.p.kmeans_initialization == "pc":
                init = "seeded-k-means++"
                seed_set = np.array([])
                print("TODO(Vijay): Not implemented")
            else:
                init = "k-means++"
                seed_set = None

            raise ValueError("multi-view spherical k-means is deprecated")