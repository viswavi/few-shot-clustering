from collections import Counter, defaultdict
import json
import jsonlines
import numpy as np
import openai
import os
from sklearn.preprocessing import normalize
import time
from tqdm import tqdm


from active_semi_clustering.exceptions import EmptyClustersException
from .constraints import preprocess_constraints
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans
from active_semi_clustering.semi_supervised.pairwise_constraints.pckmeans import PCKMeans
from active_semi_clustering.semi_supervised.pairwise_constraints.gptclustering_prompts import select_keyphrase_expansion_prompt
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

import sys
sys.path.append("cmvc")
from cmvc.helper import invertDic
from cmvc.metrics import pairwiseMetric, calcF1
from cmvc.test_performance import cluster_test

class GPTExpansionClustering(KMeans):
    def __init__(self, X, documents, encoder_model=None, dataset_name=None, prompt=None, text_type=None, prompt_for_encoder=None, keep_original_entity=True, split=None, n_clusters=3, side_information=None, read_only=False, instruction_only=False, demonstration_only=False, cache_file_name="gpt_paraphrase_cache.jsonl"):
        self.X = X
        self.dataset_name = dataset_name
        self.documents = documents
        self.encoder_model = encoder_model
        # If a dataset is specified, then we'll automatically infer the correct encoder model to use.
        # Otherwise, a model object must be provided.
        if self.dataset_name is None:
            assert self.encoder_model is not None, "Provide an encoder model to use for keyphrase clustering"
        self.prompt = prompt
        self.text_type = text_type
        self.prompt_for_encoder = prompt_for_encoder
        self.keep_original_entity = keep_original_entity
        self.n_clusters = n_clusters
        self.side_information = side_information
        cache_file = cache_file_name
        self.instruction_only = instruction_only
        self.demonstration_only = demonstration_only
        if instruction_only:
            filename_components = cache_file.split("_cache.jsonl")
            cache_file = filename_components[0] + f"_instruction_only" + "_cache.jsonl"
        elif demonstration_only:
            filename_components = cache_file.split("_cache.jsonl")
            cache_file = filename_components[0] + f"_demonstration_only" + "_cache.jsonl"
        if os.path.exists(cache_file):
            self.cache_rows = list(jsonlines.open(cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(cache_file, mode='a', flush=True)
        else:
            self.cache_writer = jsonlines.open(cache_file, mode='r')
        self.NUM_RETRIES = 1
        self.read_only = read_only

        split_str = f"_{split}" if split else ""

    def process_sentence_punctuation(self, sentences):
        processed_sentence_set = []
        for s in sentences:
            processed_sentence_set.append(s.replace("-LRB-", "(").replace("-RRB-", ")"))
        return processed_sentence_set

    def create_template_block(self, entity_idx, text_type):
        filled_template = f"""{text_type}: "{self.documents[entity_idx]}"

Keyphrases:"""
        return filled_template

    def construct_gpt3_template(self, doc_idx, instruction_only=False, demonstration_only=False):
        if self.dataset_name is not None:
            prompt_prefix = select_keyphrase_expansion_prompt(self.dataset_name)
        else:
            assert self.prompt is not None
            prompt_prefix = self.prompt

        if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k":
            text_type = "Entity"
        elif self.dataset_name == "clinc" or self.dataset_name == "bank77":
            text_type = "Query"
        elif self.dataset_name == "tweet":
            text_type = "Tweet"
        else:
            assert self.text_type is not None
            text_type = self.text_type
        completion_block = self.create_template_block(doc_idx, text_type)
        return f"{prompt_prefix}\n\n{completion_block}"

    def fit(self, X):
        document_expansion_mapping = {}
        for row in self.cache_rows:
            document_expansion_mapping[row["entity"]] = row["expansion"]

        for doc_idx, document in tqdm(enumerate(self.documents)):
            if document not in document_expansion_mapping:
                if self.read_only:
                    continue
                template_to_fill = self.construct_gpt3_template(doc_idx, instruction_only=self.instruction_only, demonstration_only=self.demonstration_only)
                print(f"PROMPT:\n{template_to_fill}")

                failure = True
                num_retries = 0
                while failure and num_retries < self.NUM_RETRIES:
                    cache_row = None
                    try:
                        start = time.perf_counter()
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": template_to_fill},
                            ],
                        )
                        message = json.loads(str(response.choices[0]))["message"]["content"]
                        if message.startswith("Keywords:"):
                            message = message[len("Keywords:"):].strip()
                        try:
                            entity_expansions = json.loads(message)
                            print(message)
                            if not isinstance(entity_expansions, list) or not isinstance(entity_expansions[0], str):
                                failure = True
                            document_expansion_mapping[document] = entity_expansions
                            cache_row = {"entity": document, "expansion": entity_expansions}
                            self.cache_writer.write(cache_row)
                            failure = False
                        except:
                            time.sleep(0.8)
                        num_retries += 1
                        end = time.perf_counter()
                        if end - start < 1:
                            time.sleep(1 - (end - start))
                    except:
                        time.sleep(3)
        if not self.read_only:
            self.cache_writer.close()

        all_expansions = []
        for doc in self.documents:
            if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k":
                doc_expansions = [doc]
            else:
                doc_expansions = []
            if doc in document_expansion_mapping:
                doc_expansions.extend(document_expansion_mapping[doc])
            all_expansions.append(", ".join(doc_expansions))

        if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k"  or self.dataset_name == "tweet":
            model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
            expansion_embeddings = model.encode(all_expansions)
        elif self.dataset_name == "bank77":
            model = INSTRUCTOR('hkunlp/instructor-large')
            prompt = "Represent the bank purpose for intent classification: "
            expansion_embeddings = model.encode([[prompt, text] for text in all_expansions])
        elif self.dataset_name == "clinc":
            model = INSTRUCTOR('hkunlp/instructor-large')
            prompt = "Represent keyphrases for topic classification: "
            expansion_embeddings = model.encode([[prompt, text] for text in all_expansions])
        else:
            assert self.dataset_name is None, f"Dataset {self.dataset_name} not found"
            if self.prompt_for_encoder is None:
                expansion_embeddings = self.encoder_model.encode(all_expansions)
            else:
                expansion_embeddings = self.encoder_model.encode([[self.prompt_for_encoder, text] for text in all_expansions])

        a_vectors = normalize(self.X, axis=1, norm="l2")
        b_vectors = normalize(expansion_embeddings, axis=1, norm="l2")
        embeddings = np.concatenate([a_vectors, b_vectors], axis=1)

        kmeans = KMeans(self.n_clusters, max_iter=100, init="k-means++", normalize_vectors=True, split_normalization=True, split_point=np.shape(self.X)[1])
        kmeans.fit(embeddings)
        self.labels_ = [int(l) for l in kmeans.labels_]
        return self
