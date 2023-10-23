from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
from txt_process_util import processTxtRemoveStopWordTokenized


dataFileTrueTxt = "data/search_snippets/search_snippets_true_text"

file1=open(dataFileTrueTxt,"r", encoding="utf8")
lines = file1.readlines()
file1.close()

for line in lines:
  arr=line.split('\t')
  true_label=arr[0]
  text=arr[1]

file1=open("stopWords.txt","r", encoding="utf8")
stopWords1 = file1.readlines()
file1.close()

stopWs=[]
for stopWord1 in stopWords1:
 stopWord1=stopWord1.strip().lower()
 if len(stopWord1)==0:
  continue
stopWs.append(stopWord1)

stopWs=set(stopWs)

data=[]
true_labels=[]

for line in lines:
  arr=line.split('\t')
  true_label=arr[0]
  text=arr[1]
  filtered_sentence = processTxtRemoveStopWordTokenized(text, stopWs)
  data.append(text)
  true_labels.append(true_label)

true_k = np.unique(true_labels).shape[0]

vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
X = vectorizer.fit_transform(data)

svd = TruncatedSVD(20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

#print(true_k)

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=5)
km.fit(X)

#print("Completeness: %0.3f" % metrics.completeness_score(true_labels, km.labels_))
#print(km.labels_)
score = metrics.normalized_mutual_info_score(true_labels, km.labels_)
#print(score)

for label in km.labels_:
  print(label)


