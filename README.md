# okb-canonicalization

## Setup
### Pull submodules

`git submodule update --init`

### Download datasets
Download [20_newsgroups.tar.gz](https://kdd.ics.uci.edu/databases/20newsgroups/20_newsgroups.tar.gz), untar it here, then move all the subdirectories (e.g. `alt.atheism/`, `comp.windows.x/`, etc) to clustering/data/20_newsgroups/:

```mv 20_newsgroups/* clustering/data/20_newsgroups/```
