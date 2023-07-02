# okb-canonicalization

## Setup
### Pull submodules

`git submodule update --init`

### Download datasets
Download [20_newsgroups.tar.gz](https://kdd.ics.uci.edu/databases/20newsgroups/20_newsgroups.tar.gz), untar it here, then move all the subdirectories (e.g. `alt.atheism/`, `comp.windows.x/`, etc) to clustering/data/20_newsgroups/:

```mv 20_newsgroups/* clustering/data/20_newsgroups/```


## Run experiments

``python -u active_clustering.py --dataset <dataset_name> --data-path <data> --dataset-split <split> --num_clusters <n_clust>  --num_seeds 1 --normalize-vectors --split-normalization   --algorithms <method> --max-feedback-given 20000 --init k-means++``

where `<method>` is one of `KMeans, PCKMeans, KMeansCorrection`, or `GPTExpansionClustering`.
