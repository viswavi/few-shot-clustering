# Few Shot Clustering

## Dependencies
```
    "scikit-learn",
    "matplotlib",
    "torch",
    "numpy",
    "openai",
    "sentence_transformers",
    "jsonlines",
    "ortools",
    "tqdm"
```

## Setup
### Pull submodules

`git submodule update --init`

# Run LLM-based clustering algorithms
`cd clustering` before running the code below.

## LLM Pairwise Constraint Clustering
Here's an example of how to run pairwise constraint clustering using an LLM to generate the constraints, from scratch, on the CLINC dataset.
First, write a prompt for generating pairwise constraints:
```
prompt = """You are tasked with clustering queries for a task-oriented dialog system based on whether they express the same general user intent. To do this, you will be given pairs of user queries and asked if they express the same general user need or intent.

Your task will be considered successful if the queries are clustered into groups that consistently express the same general intent.

Utterance #1: what's the spanish word for pasta
Utterance #2: how would they say butter in zambia

Given this context, do utterance #1 and utterance #2 likely express the same general intent? Yes

Utterance #1: roll those dice once
Utterance #2: can you roll an eight sided die and tell me what it comes up as

Given this context, do utterance #1 and utterance #2 likely express the same general intent? No

Utterance #1: how soon milk expires
Utterance #2: can you roll an eight sided die and tell me what it comes up as

Given this context, do utterance #1 and utterance #2 likely express the same general intent? Yes

Utterance #1: nice seeing you bye
Utterance #2: what was the date of my last car appointment

Given this context, do utterance #1 and utterance #2 likely express the same general intent? No"""
```


Now, use this prompt to call the OpenAI API and create clusters (note that you'll need to set your `OPENAI_API_KEY` before doing this step).
```
from wrappers import LLMPairwiseClustering

from dataloaders import load_clinc

features, labels, documents = load_clinc()

prompt_suffix = "express the same general intent?"
text_type = "Utterance"

cluster_assignments, constraints = LLMPairwiseClustering(features, documents, 150, prompt, text_type, prompt_suffix, max_feedback_given=100, pckmeans_w=0.4, cache_file="/tmp/clinc_cache_file.json", constraint_selection_algorithm="SimilarityFinder", kmeans_init="k-means++")
```

## LLM Keyphrase Expansion Clustering
We'll again show a from-scratch example on CLINC.
First, write a prompt for generating keyphrases for each datapoint:
```
prompt = """I am trying to cluster task-oriented dialog system queries based on whether they express the same general user intent. To help me with this, for a given user query, provide a comprehensive set of keyphrases that could describe this query's intent. These keyphrases should be distinct from those that might describe queries with different intents. Generate the set of keyphrases as a JSON-formatted list.

Query: "how would you say fly in italian"

Keyphrases: ["translation", "translate"]

Query: "what does assiduous mean"

Keyphrases: ["definition", "define"]

Query: "find my cellphone for me!"

Keyphrases: ["location", "find", "locate", "tracking", "track"]"""
```


Now we can call the OpenAI API (after setting `OPENAI_API_KEY`) to generate keyphrases and create clusters:
```
from wrappers import LLMKeyphraseClustering
from InstructorEmbedding import INSTRUCTOR

from dataloaders import load_clinc

features, labels, documents = load_clinc()

prompt_suffix = "express the same general intent?"
text_type = "Query"
encoder_model = INSTRUCTOR('hkunlp/instructor-large')

cluster_assignments = LLMKeyphraseClustering(features, documents, 150, prompt, text_type, encoder_model=encoder_model, prompt_for_encoder="Represent keyphrases for topic classification:", cache_file="/tmp/clinc_expansion_cache_file.json")

from utils.metric import cluster_acc
import numpy as np
print(f"Accuracy: {cluster_acc(np.array(cluster_assignments), np.array(labels))}")
```
