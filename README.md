# Few Shot Clustering

## Setup
You can either install a wheel via Pip or install from source.

### Install via pip
```
pip install few-shot-clustering
```

### Install from Source:
```
git submodule update --init
pip install -e .
```

## Other dependencies
This repository also requires `torch` if you use the Keyphrase Clustering method. This is not currently included in the pip installation for users to install custom Torch packages on their own machine/GPU, but this code was tested with:
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

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

# Run LLM-based clustering algorithms
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
from few_shot_clustering.wrappers import LLMPairwiseClustering

from few_shot_clustering.dataloaders import load_clinc

# You can provide an optional file to cache the extracted features, 
# since these are a bit expensive to compute. Example:
# cache_path = "/tmp/clinc_feature_cache.pkl"
#
# This is not necessary, as shown below.

cache_path = None
features, labels, documents = load_clinc(cache_path)

prompt_suffix = "express the same general intent?"
text_type = "Utterance"

cluster_assignments, constraints = LLMPairwiseClustering(features, documents, 150, prompt, text_type, prompt_suffix, max_feedback_given=10000, pckmeans_w=0.01, cache_file="/tmp/clinc_cache_file.json", constraint_selection_algorithm="SimilarityFinder", kmeans_init="k-means++")

from few_shot_clustering.eval_utils import cluster_acc
import numpy as np
print(f"Accuracy: {cluster_acc(np.array(cluster_assignments), np.array(labels))}")
```

In this example run, I specified 10000 examples of pairwise constraint feedback, which may take an hour to run (due to latency from the OpenAI API). Feel free to choose a smaller number, with the understanding that this may lead to reduced performance.


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
from few_shot_clustering.wrappers import LLMKeyphraseClustering
from InstructorEmbedding import INSTRUCTOR

from few_shot_clustering.dataloaders import load_clinc

# You can provide an optional file to cache the extracted features, 
# since these are a bit expensive to compute. Example:
# cache_path = "/tmp/clinc_feature_cache.pkl"
#
# This is not necessary, as shown below.

cache_path = None
features, labels, documents = load_clinc(cache_path)

prompt_suffix = "express the same general intent?"
text_type = "Query"
encoder_model = INSTRUCTOR('hkunlp/instructor-large')

cluster_assignments = LLMKeyphraseClustering(features, documents, 150, prompt, text_type, encoder_model=encoder_model, prompt_for_encoder="Represent keyphrases for topic classification:", cache_file="/tmp/clinc_expansion_cache_file.json")

from few_shot_clustering.eval_utils import cluster_acc
import numpy as np
print(f"Accuracy: {cluster_acc(np.array(cluster_assignments), np.array(labels))}")
```

## Citation
Found this useful? Please cite
```
@misc{few-shot-clustering,
      title={Large Language Models Enable Few-Shot Clustering}, 
      author={Vijay Viswanathan and Kiril Gashteovski and Carolin Lawrence and Tongshuang Wu and Graham Neubig},
      year={2023},
      eprint={2307.00524},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
