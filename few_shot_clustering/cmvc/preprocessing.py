'''
Side Information Acquisition module
'''

from helper import *
import pdb, itertools
from nltk.corpus import stopwords
from cmvc_utils import *
import pickle

'''*************************************** INPUT CLASS ********************************************'''

class SideInfo(object):
    def __init__(self, args, triples_list):
        self.p = args
        self.file = open(self.p.out_path + '/side_info.txt', 'w')
        self.triples = triples_list

        self.initVariables()
        self.process()

    def process(self):
        self.folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/'
        if not os.path.exists(self.folder_to_make):
            print('folder_to_make:', self.folder_to_make)
            os.makedirs(self.folder_to_make)
        fname1, fname2, fname3 = self.folder_to_make + '/self.rel_list', self.folder_to_make + '/self.ent_list', self.folder_to_make + '/self.sub_list'
        fname4, fname5, fname6 = self.folder_to_make + '/self.clean_ent_list', self.folder_to_make + '/self.ent2id', self.folder_to_make + '/self.rel2id'
        fname7, fname8, fname9 = self.folder_to_make + '/self.isSub', self.folder_to_make + '/self.ent_freq', self.folder_to_make + '/self.rel_freq'
        fname10, fname11, fname12 = self.folder_to_make + '/self.id2ent', self.folder_to_make + '/self.id2rel', self.folder_to_make + '/self.trpIds'
        fname13, fname14, fname15 = self.folder_to_make + '/self.sub2id', self.folder_to_make + '/self.id2sub', self.folder_to_make + '/self.obj2id'
        fname16, fname17, fname18 = self.folder_to_make + '/self.id2obj', self.folder_to_make + '/self.ent_id2sentence_list', self.folder_to_make + '/self.sentence_list'
        fname19, fname20 = self.folder_to_make + '/self.ent2triple_id_list', self.folder_to_make + '/self.rel2triple_id_list'

        if not checkFile(fname1) or not checkFile(fname2):
            print('generate side_info')
            ent1List, relList, ent2List = [], [], []  # temp variables
            self.sentence_List = []
            self.ent2triple_id_list, self.rel2triple_id_list = dict(), dict()
            triple2sentence = dict()
            if self.p.use_assume:
                self.triple_str = str('triple')
                print('use assume...')
            else:
                self.triple_str = str('triple_unique')
                print('do not use assume...')
            triple_num, sentence_num = 0, 0
            for triple in self.triples:  # Get all subject, objects and relations
                sub, rel, obj = triple[self.triple_str][0], triple[self.triple_str][1], triple[self.triple_str][2]
                ent1List.append(sub)
                relList.append(rel)
                ent2List.append(obj)
                triple2sentence[triple_num] = []

                if sub not in self.ent2triple_id_list:
                    self.ent2triple_id_list.update({sub: [triple_num]})
                else:
                    self.ent2triple_id_list[sub].append(triple_num)

                if rel not in self.rel2triple_id_list:
                    self.rel2triple_id_list.update({rel: [triple_num]})
                else:
                    self.rel2triple_id_list[rel].append(triple_num)
                
                if obj not in self.ent2triple_id_list:
                    self.ent2triple_id_list.update({obj: [triple_num]})
                else:
                    self.ent2triple_id_list[obj].append(triple_num)

                for sentence in triple['src_sentences']:
                    if self.p.replace_h:
                        sentence = sentence.replace(str(triple[self.triple_str][0]), '')
                    sentence_ = word_tokenize(sentence)
                    sentence = str()
                    for i in range(len(sentence_)):
                        w = sentence_[i]
                        if self.p.sentence_delete_stopwords:
                            if w not in stopwords.words('english'):
                                sentence += str(w)
                        else:
                            sentence += str(w)
                        if not i == len(sentence_) - 1:
                            sentence += ' '
                    # print('sentence：', type(sentence), len(sentence), sentence)
                    if len(sentence) == 0:
                        sentence += str(triple[self.triple_str][0])
                    # print('sentence：', type(sentence), len(sentence), sentence)
                    self.sentence_List.append(sentence)
                    triple2sentence[triple_num].append(sentence_num)
                    if len(self.sentence_List) == 0:
                        self.sentence_List.append(triple[self.triple_str][0])
                        # self.sentence_List.append(triple[self.triple_str][2])
                    sentence_num += 1
                triple_num += 1

            print('relList:', len(relList))  # 35812
            print('ent1List:', len(ent1List))  # 35812
            print('ent2List:', len(ent2List))  # 35812
            print('sentence_List:', len(self.sentence_List))  # 93934
            print('triple2sentence:', len(triple2sentence))  # 35812
            assert len(triple2sentence) == len(relList)

            assume_rel, assume_sub, assume_obj = dict(), dict(), dict()
            for i in range(len(relList)):
                rel = relList[i]
                if rel in assume_rel.keys():
                    assume_rel[rel].append(i)
                else:
                    assume_rel[rel] = [i]

            for i in range(len(ent1List)):
                sub = ent1List[i]
                if sub in assume_sub.keys():
                    assume_sub[sub].append(i)
                else:
                    assume_sub[sub] = [i]

            for i in range(len(ent2List)):
                obj = ent2List[i]
                if obj in assume_obj.keys():
                    assume_obj[obj].append(i)
                else:
                    assume_obj[obj] = [i]

            print('assume_rel, assume_sub, assume_obj:', len(assume_rel), len(assume_sub),
                  len(assume_obj))  # 18288 12295 14935
            self.rel_list = list(assume_rel.keys())
            self.sub_list = list(assume_sub.keys())
            self.obj_list = list(assume_obj.keys())
            self.ent_list = []  # self.ent_list 's order is self.sub_list + self.obj_list
            self.ent_id2sentence_list = []
            # print('assume_sub:', assume_sub)  # {'The Guardian': [0, 1], 'Guardian': [2], 'Franz Kafka': [3, 4], 'Kafka': [5],

            for i in range(len(self.sub_list)):
                ent = self.sub_list[i]
                ids = assume_sub[ent]
                self.ent_id2sentence_list.append([])
                self.ent_list.append(ent)
                for id in ids:
                    self.ent_id2sentence_list[i] += triple2sentence[id]

            for i in range(len(self.obj_list)):
                obj = self.obj_list[i]
                ids = assume_obj[obj]
                if obj in self.sub_list:
                    continue
                else:
                    self.ent_list.append(obj)
                    self.ent_id2sentence_list.append([])
                    index = len(self.ent_id2sentence_list) - 1
                    for id in ids:
                        self.ent_id2sentence_list[index] += triple2sentence[id]

            print('self.ent_id2sentence_list:', len(self.ent_id2sentence_list))  # 23735
            print('self.ent_list:', len(self.ent_list))  # 23735
            print('self.sub_list:', len(self.sub_list))  # 12295
            print('self.obj_list:', len(self.obj_list))  # 14935
            print('self.rel_list:', len(self.rel_list))  # 18288

            # Generate a unique id for each entity and relations
            self.ent2id = dict([(v, k) for k, v in enumerate(self.ent_list)])
            self.rel2id = dict([(v, k) for k, v in enumerate(self.rel_list)])
            self.sub2id = dict([(v, k) for k, v in enumerate(self.sub_list)])
            self.obj2id = dict([(v, k) for k, v in enumerate(self.obj_list)])
            print('self.sub2id:', len(self.sub2id))  # 12295
            print('self.obj2id:', len(self.obj2id))  # 14935
            print('self.ent2id:', len(self.ent2id))  # 23735
            print('self.rel2id:', len(self.rel2id))  # 18288

            self.isSub = {}
            for sub in self.sub_list:
                self.isSub[self.ent2id[sub]] = 1
            print('self.isSub:', len(self.isSub))  # 12295

            # Get frequency of occurence of entities and relations
            for ele in ent1List:
                ent = self.ent2id[ele]
                self.ent_freq[ent] = self.ent_freq.get(ent, 0)
                self.ent_freq[ent] += 1

            for ele in ent2List:
                ent = self.ent2id[ele]
                self.ent_freq[ent] = self.ent_freq.get(ent, 0)
                self.ent_freq[ent] += 1

            for ele in relList:
                rel = self.rel2id[ele]
                self.rel_freq[rel] = self.rel_freq.get(rel, 0)
                self.rel_freq[rel] += 1

            # Creating inverse mapping as well
            self.id2ent = invertDic(self.ent2id)
            self.id2rel = invertDic(self.rel2id)
            self.id2sub = invertDic(self.sub2id)
            self.id2obj = invertDic(self.obj2id)
            # self.id2text = invertDic(self.text2id)

            print('self.ent_freq:', len(self.ent_freq))  # 23735
            print('self.rel_freq:', len(self.rel_freq))  # 18288
            print('self.id2ent:', len(self.id2ent))  # 23735
            print('self.id2rel:', len(self.id2rel))  # 18288
            print('self.id2sub:', len(self.id2sub))  # 12295
            print('self.id2obj:', len(self.id2obj))  # 14935

            for triple in self.triples:
                trp = (
                    self.ent2id[triple[self.triple_str][0]], self.rel2id[triple[self.triple_str][1]],
                    self.ent2id[triple[self.triple_str][2]])
                self.trpIds.append(trp)
            print('self.trpIds:', len(self.trpIds))  # 35812

            pickle.dump(self.rel_list, open(fname1, 'wb'))
            pickle.dump(self.ent_list, open(fname2, 'wb'))
            pickle.dump(self.sub_list, open(fname3, 'wb'))
            pickle.dump(self.obj_list, open(fname4, 'wb'))
            pickle.dump(self.ent2id, open(fname5, 'wb'))
            pickle.dump(self.rel2id, open(fname6, 'wb'))
            pickle.dump(self.isSub, open(fname7, 'wb'))
            pickle.dump(self.ent_freq, open(fname8, 'wb'))
            pickle.dump(self.rel_freq, open(fname9, 'wb'))
            pickle.dump(self.id2ent, open(fname10, 'wb'))
            pickle.dump(self.id2rel, open(fname11, 'wb'))
            pickle.dump(self.trpIds, open(fname12, 'wb'))
            pickle.dump(self.sub2id, open(fname13, 'wb'))
            pickle.dump(self.id2sub, open(fname14, 'wb'))
            pickle.dump(self.obj2id, open(fname15, 'wb'))
            pickle.dump(self.id2obj, open(fname16, 'wb'))
            pickle.dump(self.ent_id2sentence_list, open(fname17, 'wb'))
            pickle.dump(self.sentence_List, open(fname18, 'wb'))
            pickle.dump(self.ent2triple_id_list, open(fname19, 'wb'))
            pickle.dump(self.rel2triple_id_list, open(fname20, 'wb'))

        else:
            print('load side_info')
            self.rel_list = pickle.load(open(fname1, 'rb'))
            self.ent_list = pickle.load(open(fname2, 'rb'))
            self.sub_list = pickle.load(open(fname3, 'rb'))
            self.obj_list = pickle.load(open(fname4, 'rb'))
            self.ent2id = pickle.load(open(fname5, 'rb'))
            self.rel2id = pickle.load(open(fname6, 'rb'))
            self.isSub = pickle.load(open(fname7, 'rb'))
            self.ent_freq = pickle.load(open(fname8, 'rb'))
            self.rel_freq = pickle.load(open(fname9, 'rb'))
            self.id2ent = pickle.load(open(fname10, 'rb'))
            self.id2rel = pickle.load(open(fname11, 'rb'))
            self.trpIds = pickle.load(open(fname12, 'rb'))
            self.sub2id = pickle.load(open(fname13, 'rb'))
            self.id2sub = pickle.load(open(fname14, 'rb'))
            self.obj2id = pickle.load(open(fname15, 'rb'))
            self.id2obj = pickle.load(open(fname16, 'rb'))
            self.ent_id2sentence_list = pickle.load(open(fname17, 'rb'))
            self.sentence_List = pickle.load(open(fname18, 'rb'))
            self.ent2triple_id_list = pickle.load(open(fname19, 'rb'))
            self.rel2triple_id_list = pickle.load(open(fname20, 'rb'))

        print('self.rel_list:', type(self.rel_list), len(self.rel_list))
        print('self.ent_list:', type(self.ent_list), len(self.ent_list))
        print('self.sub_list:', type(self.sub_list), len(self.sub_list))
        print('self.obj_list:', type(self.obj_list), len(self.obj_list))
        print('self.ent2id:', type(self.ent2id), len(self.ent2id))
        print('self.rel2id:', type(self.rel2id), len(self.rel2id))
        print('self.isSub:', type(self.isSub), len(self.isSub))
        print('self.ent_freq:', type(self.ent_freq), len(self.ent_freq))
        print('self.rel_freq:', type(self.rel_freq), len(self.rel_freq))
        print('self.id2ent:', type(self.id2ent), len(self.id2ent))
        print('self.id2rel:', type(self.id2rel), len(self.id2rel))
        print('self.trpIds:', type(self.trpIds), len(self.trpIds))
        print('self.sub2id:', type(self.sub2id), len(self.sub2id))
        print('self.id2sub:', type(self.id2sub), len(self.id2sub))
        print('self.obj2id:', type(self.obj2id), len(self.obj2id))
        print('self.id2obj:', type(self.id2obj), len(self.id2obj))
        print('self.ent_id2sentence_list:', type(self.ent_id2sentence_list), len(self.ent_id2sentence_list))
        print('self.sentence_List:', type(self.sentence_List), len(self.sentence_List))
        print('self.ent2triple_id_list:', type(self.ent2triple_id_list), len(self.ent2triple_id_list))
        print('self.rel2triple_id_list:', type(self.rel2triple_id_list), len(self.rel2triple_id_list))
        print()

        if self.p.use_Entity_linking_dict:
            fname1 = '../file/Entity_linking_dict'
            if not checkFile(fname1):
                print('generate Entity_linking_dict')
                self.Entity_linking_dict = self.generate_Entity_linking_dict()
                pickle.dump(self.Entity_linking_dict, open(fname1, 'wb'))
                print('Entity_linking_dict :', len(self.Entity_linking_dict), type(self.Entity_linking_dict))
            else:
                print('load Entity_linking_dict')
                self.Entity_linking_dict = pickle.load(open(fname1, 'rb'))
                print('Entity_linking_dict :', len(self.Entity_linking_dict), type(self.Entity_linking_dict))
            fname1, fname2 = self.folder_to_make + '/look_up_entity_EL_dict', self.folder_to_make + '/look_up_relation_EL_dict'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate look_up Entity_linking_dict')
                self.look_up_entity_EL_dict, self.look_up_relation_EL_dict = self.look_up_Entity_linking_dict()
                pickle.dump(self.look_up_entity_EL_dict, open(fname1, 'wb'))
                pickle.dump(self.look_up_relation_EL_dict, open(fname2, 'wb'))
            else:
                print('load look_up Entity_linking_dict')
                self.look_up_entity_EL_dict = pickle.load(open(fname1, 'rb'))
                self.look_up_relation_EL_dict = pickle.load(open(fname2, 'rb'))

            fname1, fname2 = self.folder_to_make + '/entity_score_EL_dict', self.folder_to_make + '/relation_score_EL_dict'
            if not checkFile(fname1) or not checkFile(fname2) or self.p.change_EL_threshold:
                print('generate max score Entity_linking_dict')
                self.entity_score_EL_dict, self.relation_score_EL_dict = self.score_Entity_linking_dict()
                pickle.dump(self.entity_score_EL_dict, open(fname1, 'wb'))
                pickle.dump(self.relation_score_EL_dict, open(fname2, 'wb'))
            else:
                print('load max score Entity_linking_dict')
                self.entity_score_EL_dict = pickle.load(open(fname1, 'rb'))
                self.relation_score_EL_dict = pickle.load(open(fname2, 'rb'))


            fname1, fname2 = self.folder_to_make + '/ent_old_id2new_id', self.folder_to_make + '/rel_old_id2new_id'
            if not checkFile(fname1) or self.p.change_EL_threshold:
                print('generate el ent_old_id2new_id')
                self.ent_old_id2new_id, self.rel_old_id2new_id = self.generate_old_id2new_id()

                pickle.dump(self.ent_old_id2new_id, open(fname1, 'wb'))
                pickle.dump(self.rel_old_id2new_id, open(fname2, 'wb'))

            else:
                print('load new ent2id_dict')
                self.ent_old_id2new_id = pickle.load(open(fname1, 'rb'))
                self.rel_old_id2new_id = pickle.load(open(fname2, 'rb'))

            fname1, fname2 = self.folder_to_make + '/new_seed_sim', self.folder_to_make + '/new_seed_trpIds'
            if not checkFile(fname1) or not checkFile(fname2) or self.p.change_EL_threshold:
                print('generate EL dict seed')
                self.seed_sim, self.seed_trpIds = self.get_EL_seed()

                pickle.dump(self.seed_sim, open(fname1, 'wb'))
                pickle.dump(self.seed_trpIds, open(fname2, 'wb'))
            else:
                print('load seed')
                self.seed_sim = pickle.load(open(fname1, 'rb'))
                self.seed_trpIds = pickle.load(open(fname2, 'rb'))

    def generate_Entity_linking_dict(self):
        Entity_linking_dict = dict()
        with open(self.p.Entity_linking_dict_loc, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split('\t')  # 去掉换行符\n, 将每一行以空格为分隔符转换成列表
                key, value = line[0], line[1:len(line)]
                Entity_linking_dict.update({key: value})
        return Entity_linking_dict

    def look_up_Entity_linking_dict(self):
        look_up_entity_EL_dict, look_up_relation_EL_dict = dict(), dict()
        num1, num2, num3, num4 = 0, 0, 0, 0
        print('num of entity', len(self.ent_list), 'num of relation:', len(self.rel_list))
        for i in range(len(self.ent_list)):
            if self.Entity_linking_dict.__contains__(self.ent_list[i]):
                value = self.Entity_linking_dict[self.ent_list[i]]
                look_up_entity_EL_dict.update({self.ent_list[i]: value})
            else:
                # print('not have1', self.ent_list[i])
                num1 += 1
                lower_key = str(self.ent_list[i]).lower()
                if self.Entity_linking_dict.__contains__(lower_key):
                    value = self.Entity_linking_dict[lower_key]
                    look_up_entity_EL_dict.update({self.ent_list[i]: value})
                else:
                    # print('not have2', self.ent_list[i])
                    num2 += 1
        for i in range(len(self.rel_list)):
            if self.Entity_linking_dict.__contains__(self.rel_list[i]):
                value = self.Entity_linking_dict[self.rel_list[i]]
                look_up_relation_EL_dict.update({self.rel_list[i]: value})
            else:
                num3 += 1
                lower_key = str(self.rel_list[i]).lower()
                if self.Entity_linking_dict.__contains__(lower_key):
                    value = self.Entity_linking_dict[lower_key]
                    look_up_relation_EL_dict.update({self.rel_list[i]: value})
                else:
                    num4 += 1
        print('num1:', num1, 'num2:', num2)
        print('look_up_entity_EL_dict:', len(look_up_entity_EL_dict), type(look_up_entity_EL_dict))
        print('num3:', num3, 'num4:', num4)
        print('look_up_relation_EL_dict:', len(look_up_relation_EL_dict), type(look_up_relation_EL_dict))
        return look_up_entity_EL_dict, look_up_relation_EL_dict

    def score_Entity_linking_dict(self):
        self.entity_score_EL_dict, self.relation_score_EL_dict = dict(), dict()
        print('self.p.entity_EL_threshold:', self.p.entity_EL_threshold)  # 0
        print('self.p.relation_EL_threshold:', self.p.relation_EL_threshold)  # 0

        for mention, entity in self.look_up_entity_EL_dict.items():
            score_sum, score_list = 0, []
            for i in range(len(entity)):
                if i % 2 == 0: continue
                else: score_sum += int(entity[i])
            for i in range(len(entity)):
                if i % 2 == 0: continue
                else:
                    if score_sum == 0:score_sum=1
                    score = int(entity[i]) / score_sum
                    entity[i] = score
                    score_list.append(score)
            max_score = max(score_list)
            max_score_index = entity.index(max_score)
            if max_score > self.p.entity_EL_threshold:
                if self.entity_score_EL_dict.__contains__(entity[max_score_index - 1]):
                    self.entity_score_EL_dict[entity[max_score_index - 1]].append(mention)
                else:
                    self.entity_score_EL_dict.update({entity[max_score_index - 1]: [mention]})

        for mention, entity in self.look_up_relation_EL_dict.items():
            score_sum, score_list = 0, []
            for i in range(len(entity)):
                if i % 2 == 0: continue
                else: score_sum += int(entity[i])
            for i in range(len(entity)):
                if i % 2 == 0: continue
                else:
                    if score_sum == 0:score_sum = 1
                    score = int(entity[i]) / score_sum
                    entity[i] = score
                    score_list.append(score)
            max_score = max(score_list)
            max_score_index = entity.index(max_score)
            if max_score > self.p.entity_EL_threshold:
                if self.relation_score_EL_dict.__contains__(entity[max_score_index - 1]):
                    self.relation_score_EL_dict[entity[max_score_index - 1]].append(mention)
                else:
                    self.relation_score_EL_dict.update({entity[max_score_index - 1]: [mention]})
        return self.entity_score_EL_dict, self.relation_score_EL_dict

    def generate_old_id2new_id(self):
        ent_old_id2new_id, rel_old_id2new_id = {}, {}
        for entity, mention in self.entity_score_EL_dict.items():
            max_len, max_len_of_ent = 0, str()
            for i in range(len(mention)):
                ent = mention[i]
                if len(mention) > max_len:
                    max_len = len(mention)
                    max_len_of_ent = ent
            for i in range(len(mention)):
                ent = mention[i]
                ent_old_id2new_id.update({self.ent2id[ent]: self.ent2id[max_len_of_ent]})
        for ent in self.ent_list:
            if self.ent2id[ent] in ent_old_id2new_id.keys(): continue
            else:
                ent_old_id2new_id.update({self.ent2id[ent]: self.ent2id[ent]})

        for entity, mention in self.relation_score_EL_dict.items():
            max_len, max_len_of_rel = 0, str()
            for i in range(len(mention)):
                rel = mention[i]
                if len(mention) > max_len:
                    max_len = len(mention)
                    max_len_of_rel = rel
            for i in range(len(mention)):
                rel = mention[i]
                rel_old_id2new_id.update({self.rel2id[rel]: self.rel2id[max_len_of_rel]})
        for rel in self.rel_list:
            if self.rel2id[rel] in rel_old_id2new_id.keys(): continue
            else:
                rel_old_id2new_id.update({self.rel2id[rel]: self.rel2id[rel]})

        return ent_old_id2new_id, rel_old_id2new_id

    def get_EL_seed(self):
        seed_sim, seed_trpIds = [], []

        fname1, fname2 = self.folder_to_make + '/1E_init', self.folder_to_make + '/1R_init'
        if not checkFile(fname1) or not checkFile(fname2):
            print('generate pre-trained embeddings')
            import gensim
            model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
            self.E_init = getEmbeddings(model, self.ent_list, self.p.embed_dims)
            self.R_init = getEmbeddings(model, self.rel_list, self.p.embed_dims)

            pickle.dump(self.E_init, open(fname1, 'wb'))
            pickle.dump(self.R_init, open(fname2, 'wb'))
        else:
            print('load pre-trained embeddings')
            self.E_init = pickle.load(open(fname1, 'rb'))
            self.R_init = pickle.load(open(fname2, 'rb'))

        for i in range(len(self.ent_list)):
            ent1= self.ent_list[i]
            old_id1 = self.ent2id[ent1]
            for j in range(i + 1, len(self.ent_list)):
                ent2 = self.ent_list[j]
                old_id2 = self.ent2id[ent2]
                new_id1, new_id2 = self.ent_old_id2new_id[old_id1], self.ent_old_id2new_id[old_id2]
                if new_id1 == new_id2:
                    if not np.dot(self.E_init[i], self.E_init[j]) == 0:sim = cos_sim(self.E_init[i], self.E_init[j])
                    else:sim = 0
                    for ent in [ent1, ent2]:
                        triple_list = self.ent2triple_id_list[ent]
                        for triple_id in triple_list:
                            triple = self.trpIds[triple_id]
                            if str(self.id2ent[triple[0]]) == str(ent1):
                                trp = (self.ent2id[str(ent2)], triple[1], triple[2])
                                seed_trpIds.append(trp)
                                seed_sim.append(sim)
                            if str(self.id2ent[triple[0]]) == str(ent2):
                                trp = (self.ent2id[str(ent1)], triple[1], triple[2])
                                seed_trpIds.append(trp)
                                seed_sim.append(sim)
                            if str(self.id2ent[triple[2]]) == str(ent1):
                                trp = (triple[0], triple[1], self.ent2id[str(ent2)])
                                seed_trpIds.append(trp)
                                seed_sim.append(sim)
                            if str(self.id2ent[triple[2]]) == str(ent2):
                                trp = (triple[0], triple[1], self.ent2id[str(ent1)])
                                seed_trpIds.append(trp)
                                seed_sim.append(sim)
        entity_seed_length = len(seed_sim)
        for i in range(len(self.rel_list)):
            rel1= self.rel_list[i]
            old_id1 = self.rel2id[rel1]
            for j in range(i + 1, len(self.rel_list)):
                rel2 = self.rel_list[j]
                old_id2 = self.rel2id[rel2]
                new_id1, new_id2 = self.rel_old_id2new_id[old_id1], self.rel_old_id2new_id[old_id2]
                if new_id1 == new_id2:
                    if not np.dot(self.R_init[i], self.R_init[j]) == 0:sim = cos_sim(self.R_init[i], self.R_init[j])
                    else:sim = 0
                    for rel in [rel1, rel2]:
                        triple_list = self.rel2triple_id_list[rel]
                        for triple_id in triple_list:
                            triple = self.trpIds[triple_id]
                            if str(self.id2rel[triple[1]]) == str(rel1):
                                trp = (triple[0], self.rel2id[str(rel2)], triple[2])
                                seed_trpIds.append(trp)
                                seed_sim.append(sim)
                            if str(self.id2rel[triple[1]]) == str(rel2):
                                trp = (triple[0], self.rel2id[str(rel1)], triple[2])
                                seed_trpIds.append(trp)
                                seed_sim.append(sim)
        relation_seed_length = len(seed_sim) - entity_seed_length
        print('entity_seed_length:', entity_seed_length, 'relation_seed_length:', relation_seed_length, 'all_seed_length:', len(seed_sim))
        return seed_sim, seed_trpIds


    ''' ATTRIBUTES DECLARATION '''

    def initVariables(self):
        self.ent_list = None  # List of all entities
        self.clean_ent_list = []
        self.rel_list = None  # List of all relations
        self.trpIds = []  # List of all triples in id format
        self.node = []
        self.seed_trpIds = []
        self.new_trpIds = []

        self.ent2id = None  # Maps entity to its id (o2o)
        self.rel2id = None  # Maps relation to its id (o2o)
        self.id2ent = None  # Maps id to entity (o2o)
        self.id2rel = None  # Maps id to relation (o2o)

        self.ent_freq = {}  # Entity to its frequency
        self.rel_freq = {}  # Relation to its frequency

        self.ent2name_seed = {}
        self.rel2name_seed = {}