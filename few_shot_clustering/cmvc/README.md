<a name="readme-top"></a>
# Multi-View Clustering for Open Knowledge Base Canonicalization

Source code and data sets for the [SIGKDD 2022](https://kdd.org/kdd2022/) paper: [Multi-View Clustering for Open Knowledge Base Canonicalization](https://dl.acm.org/doi/pdf/10.1145/3534678.3539449)

### Dependencies

* Compatible with Python 3.6
* Dependencies can be installed using requirements.txt
* Please download the init_dict folder from this webpage: 
https://drive.google.com/file/d/17xYnisHhpFcYwHgF3-kYquw2ZZqV9-3Y/view?usp=sharing
* Please download the crawl-300d-2M.vec.zip from https://fasttext.cc/docs/en/english-vectors.html into init_dict. 

### Data sets
* Please download the ReVerb45K and OPIEC59K data sets from this webpage: 
https://drive.google.com/file/d/12naKrTctiV1O2e5chdMU2CTUzZVKMgp5/view?usp=sharing

* Please download the NYTimes2018 data set from the webpage of its original authors: 
https://heathersherry.github.io/ICDE2019_data.html

### Usage

##### Run the main code:

* python CMVC_main_reverb45k.py
* python CMVC_main_NYT.py
* python CMVC_main_opiec.py

### Contact

Yang Yang (`y2@mail.nankai.edu.cn`)

### Citation
Please cite the following paper if you use this code in your work. 

```bibtex
@inproceedings{shen2022multi,
  title={Multi-View Clustering for Open Knowledge Base Canonicalization},
  author={Shen, Wei and Yang, Yang and Liu, Yinan},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1578--1588},
  year={2022}
}
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
