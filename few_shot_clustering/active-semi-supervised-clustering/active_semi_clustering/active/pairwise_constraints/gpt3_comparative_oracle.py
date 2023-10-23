import json
import jsonlines
import math
import os
import time

import openai

from .example_oracle import MaximumQueriesExceeded

class GPT3ComparativeOracle:
    def __init__(self, X, labels, dataset_name, split=None, max_queries_cnt=2500, num_predictions=5, side_information=None, read_only=True):
        self.labels = labels
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt
        self.num_predictions = num_predictions

        self.side_information = side_information
        self.cache_dir = "/projects/ogma1/vijayv/okb-canonicalization/clustering/file/gpt3_cache"
        self.dataset_name = dataset_name
        self.cache_file = os.path.join(self.cache_dir, f"{dataset_name}_triplet_comparison_cache.jsonl")
        if os.path.exists(self.cache_file):
            self.cache_rows = list(jsonlines.open(self.cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(self.cache_file, mode='a', flush=True)
        else:
            self.cache_writer = jsonlines.open(self.cache_file, mode='r')

        self.NUM_RETRIES = 2
        self.read_only = read_only

        side_info = self.side_information.side_info
        self.sentence_unprocessing_mapping_file = os.path.join(self.cache_dir, f"{dataset_name}_{split}_sentence_unprocessing_map.json")
        sentence_unprocessing_mapping = json.load(open(self.sentence_unprocessing_mapping_file))
        selected_sentences = []
        ents = []

        for i in range(len(X)):
            ents.append(side_info.id2ent[i])
            entity_sentence_idxs = side_info.ent_id2sentence_list[i]
            unprocessed_sentences = [sentence_unprocessing_mapping[side_info.sentence_List[j]] for j in entity_sentence_idxs]
            entity_sentences = self.process_sentence_punctuation(unprocessed_sentences)
            entity_sentences_dedup = list(set(entity_sentences))

            '''
            Choose longest sentence under 306 characers, as in
            https://github.com/Yang233666/cmvc/blob/6e752b1aa5db7ff99eb2fa73476e392a00b0b89a/Context_view.py#L98
            '''
            longest_sentences = sorted([s for s in entity_sentences_dedup if len(s) < 599], key=len)
            selected_sentences.append(list(set(longest_sentences[:3])))

        self.ents = ents
        self.selected_sentences = selected_sentences

        self.gpt3_triplet_labels = {}
        for row in self.cache_rows:
            triple = (row["entity1"], row["entity2"], row["entity3"])
            self.gpt3_triplet_labels[triple] = row["labels"]


    def process_sentence_punctuation(self, sentences):
        processed_sentence_set = []
        for s in sentences:
            processed_sentence_set.append(s.replace("-LRB-", "(").replace("-RRB-", ")"))
        return processed_sentence_set


    def construct_single_example(self, i, j, k, true_label=None, add_label=True):
        context_labels = ["a", "b", "c", "d"]
        context_1 = "\n".join([context_labels[ind] + ") " + '"' + sent + '"' for ind, sent in enumerate(self.selected_sentences[i])])
        context_2 = "\n".join([context_labels[ind] + ") " + '"' + sent + '"' for ind, sent in enumerate(self.selected_sentences[j])])
        context_3 = "\n".join([context_labels[ind] + ") " + '"' + sent + '"' for ind, sent in enumerate(self.selected_sentences[k])])
        if self.dataset_name == "OPIEC59k":
            prompt_suffix = "link to the same entity's article on Wikipedia"
        elif self.dataset_name == "reverb45k":
            prompt_suffix = "link to the same entity on a knowledge graph like Freebase"
        else:
            raise NotImplementedError
        template_prefix = f"""1) {self.ents[i]}

Context Sentences:\n{context_1}

2) {self.ents[j]}

Context Sentence:\n{context_2}

3) {self.ents[k]}

Context Sentence:\n{context_3}

Given this context, are {self.ents[i]} and {self.ents[j]} more likely to {prompt_suffix} than {self.ents[i]} and {self.ents[k]}? """
        if add_label:
            assert true_label is True or true_label is False
            label = "Yes" if true_label is True else "No"
            full_example = template_prefix + label
            return full_example
        else:
            return template_prefix, context_1, context_2, context_3

    def construct_pairwise_oracle_prompt(self, i, j, k):
        side_info = self.side_information.side_info
        if self.dataset_name == "OPIEC59k":
            instruction = """You are tasked with clustering entity strings based on whether they refer to the same Wikipedia article. To do this, you will be given triplets of entity names and asked if the first entity, if linked to a Wikipedia article, is more likely referring to the second entity than it is to the third entity. Entity names may be truncated, abbreviated, or ambiguous.

To help you make this determination, you will be given up to three context sentences from Wikipedia where the entity is used as anchor text for a hyperlink. Amongst each set of examples for a given entity, the entity for all three sentences is a link to the same article on Wikipedia. Based on these examples, you will decide whether the first entity is more likely to to link to the same Wikipedia article as the second entity than the third entity. Respond with "Yes" or "No".

Please note that the context sentences may not be representative of the entity's typical usage, but should aid in resolving the ambiguity of entities that have similar or overlapping meanings.

To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the entities will generally be used in the same contexts, whether the context sentences mention the same topic, and whether the entities have the same domain and scope of meaning.

Your task will be considered successful if the entities are clustered into groups that consistently refer to the same Wikipedia articles."""
            example_1 = self.construct_single_example(side_info.ent2id["B.A"], side_info.ent2id["M.D."], side_info.ent2id["bachelor"], False)
            example_2 = self.construct_single_example(side_info.ent2id["Alexander I of Russia"], side_info.ent2id["Alexander I"], side_info.ent2id["Paul I of Russia"], True)
            example_3 = self.construct_single_example(side_info.ent2id["US Army Corps of Engineers"], side_info.ent2id["Electronics Engineers"], side_info.ent2id["United States Army Corps of Engineers"], False)
            prefix = "\n\n".join([example_1, example_2, example_3])
        elif self.dataset_name == "reverb45k":
            instruction = """You are tasked with clustering entity strings based on whether they link to the same entity (e.g. a concept, person, or organization) on the Freebase knowledge graph. To do this, you will be given triplets of entity names and asked if the first entity, if linked to a knowledge graph, is more likely referring to the second entity than it is to the third entity. Entity names may be truncated, abbreviated, or ambiguous.

To help you make this determination, you will be given up to three context sentences from the internet that mention each entity. Amongst each set of examples for a given entity, assume that the entity mentioned in all three context sentences links refers to the same object. Based on these examples, you will decide whether the first entity is more likely to to link to the *same* knowledge graph entity as the second entity than the third entity. Respond with "Yes" or "No".

Please note that the context sentences may not be representative of the entity's typical usage, but should aid in resolving the ambiguity of entities that have similar or overlapping meanings.

To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the entities will generally be used in the same contexts, whether the entities likely refer to the same person or organization, whether the context sentences mention the same topic, and whether the entities have the same domain and scope of meaning.

Your task will be considered successful if the entities are clustered into groups that consistently link to the same knowledge graph node."""
            example_1 = self.construct_single_example(side_info.ent2id["Hannibal"], side_info.ent2id["Hannibal Barca"], side_info.ent2id["Darius"], True)
            example_2 = self.construct_single_example(side_info.ent2id["Church"], side_info.ent2id["Lutheran Church"], side_info.ent2id["Roman Catholic Church"], False)
            example_3 = self.construct_single_example(side_info.ent2id["Charlie Williams"], side_info.ent2id["Roger Williams"], side_info.ent2id["Williams"], True)
            prefix = "\n\n".join([example_1, example_2, example_3])
        else:
            raise NotImplementedError
        filled_template, context_1, context_2, context_3 = self.construct_single_example(i, j, k, add_label = False)
        return "\n\n".join([instruction, prefix, filled_template]), context_1, context_2, context_3

    @staticmethod
    def filter_high_entropy_predictions(pair_labels, majority_class_threshold=0.999999):
        '''If the majority class probability is < `majority_class_threshold`, return None'''
        assert None not in pair_labels
        p = sum(pair_labels) / len(pair_labels)

        if p > 0.5 and p > majority_class_threshold:
            return True
        elif p < 0.5 and p < 1 - majority_class_threshold:
            return False
        else:
            return None

    def query(self, i, comparison_pair):
        j, k = comparison_pair
        print(f"Querying entity {i} against {j} and {k}")
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            triplet = (self.ents[i], self.ents[j], self.ents[k])

            if  triplet in self.gpt3_triplet_labels:
                return self.filter_high_entropy_predictions(self.gpt3_triplet_labels[triplet])
            elif self.read_only is True:
                return None

            prompt, context1, context2, context3 = self.construct_pairwise_oracle_prompt(i, j, k)
            print("PROMPT:\n" + prompt)

            triplet_labels_not_none = []

            failure = True
            num_retries = 0
            while failure and num_retries < self.NUM_RETRIES:
                cache_row = None
                try:
                    start = time.perf_counter()
                    print(f"Querying {self.ents[i]}, {self.ents[j]}, and {self.ents[k]}...")
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=1.0,
                        max_tokens=1,
                        n=self.num_predictions,
                    )

                    triplet_labels = []
                    for choice in response.choices:
                        message = json.loads(str(choice))["message"]["content"]
                        if message.strip() == "Yes":
                            pair_label = True
                        elif message.strip() == "No":
                            pair_label = False
                        else:
                            pair_label = None
                        triplet_labels.append(pair_label)

                    print(f"labels:\n{triplet_labels}\n\n")
                    triplet_labels_not_none = [x for x in triplet_labels if x is not None]
                    if len(triplet_labels_not_none) <= self.num_predictions / 2:
                        time.sleep(0.8)
                    else:
                        cache_row = {"entity1": self.ents[i],
                                     "entity2": self.ents[j],
                                     "entity3": self.ents[k],
                                     "labels": triplet_labels_not_none,
                                     "p_true": round(sum(triplet_labels_not_none) / len(triplet_labels_not_none), 4),
                                     "context1": context1,
                                     "context2": context2,
                                     "context3": context3
                                     }
                        self.cache_writer.write(cache_row)
                        self.gpt3_triplet_labels[triplet] = triplet_labels_not_none
                        failure = False


                    num_retries += 1
                    end = time.perf_counter()
                    if end - start < 1:
                        time.sleep(1 - (end - start))
                except Exception as e:
                    print(e)
                    time.sleep(3)

            if failure:
                return None
            else:
                return self.filter_high_entropy_predictions(triplet_labels_not_none)
        else:
            breakpoint()
            raise MaximumQueriesExceeded
