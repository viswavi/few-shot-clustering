import re
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import random
import sys
from time import time
from sklearn import metrics
from sklearn.linear_model import RidgeClassifier, LogisticRegression
#from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import Perceptron
#from sklearn.naive_bayes import BernoulliNB, MultinomialNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import linear_model
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sent_vecgenerator import generate_sent_vecslist_toktextdatas, generate_sent_vecs_toktextdata
from word_vec_extractor import extract_word_vecs, extract_word_vecs_list

##read labels obtained from external clustering algorithm
extClustLabelFile = "/home/owner/PhD/dr.norbert/dataset/shorttext/agnews/agnews-sparse-w2vec-alpha-8000-0-labels"
extClustLabels = readClustLabels(extClustLabelFile)


#file=open("D:/PhD/dr.norbert/dataset/shorttext/stackoverflow/semisupervised/stackoverflowraw_ensembele_train","r")
#file=open("D:/PhD/dr.norbert/dataset/shorttext/data-web-snippets/semisupervised/data-web-snippetsraw_ensembele_train","r")
#file=open("D:/PhD/dr.norbert/dataset/shorttext/biomedical/semisupervised/biomedicalraw_ensembele_train","r")
#file=open("/users/grad/rakib/dr.norbert/dataset/shorttext/agnews/semisupervised/agnewsraw_ensembele_train","r")
file=open("/home/owner/PhD/dr.norbert/dataset/shorttext/agnews/semisupervised/agnewsraw_ensembele_train","r")
lines = file.readlines()
file.close()

terms = []

train_data = []
train_labels = []
train_trueLabels = []

train_textdata = []

for line in lines:
 line=line.lower().strip() 
 arr = re.split("\t", line)
 train_data.append(arr[2])
 word_tokens = word_tokenize(arr[2])
 train_textdata.append(word_tokens)
 terms.extend(word_tokens)
 train_labels.append(arr[0])
 train_trueLabels.append(arr[1])

 
#file=open("D:/PhD/dr.norbert/dataset/shorttext/stackoverflow/semisupervised/stackoverflowraw_ensembele_test","r")  
file=open("/home/owner/PhD/dr.norbert/dataset/shorttext/agnews/semisupervised/agnewsraw_ensembele_test","r") 
#file=open("D:/PhD/dr.norbert/dataset/shorttext/data-web-snippets/semisupervised/data-web-snippetsraw_ensembele_test","r")
#file=open("D:/PhD/dr.norbert/dataset/shorttext/biomedical/semisupervised/biomedicalraw_ensembele_test","r")  

lines = file.readlines()
file.close()

test_data = []
test_labels = []

test_textdata = []

for line in lines:
 line=line.lower().strip() 
 arr = re.split("\t", line)
 word_tokens = word_tokenize(arr[2])
 test_textdata.append(word_tokens)
 terms.extend(word_tokens)
 test_data.append(arr[2])
 test_labels.append(arr[1])

terms = set(terms)
gloveFile = "/home/owner/PhD/dr.norbert/dataset/shorttext/glove.42B.300d/glove.42B.300d.txt"
file=open(gloveFile,"r")
vectorlines = file.readlines()
file.close()

lineProgCount = 0
termsVectors = []

for vecline in vectorlines:
 vecarr = vecline.strip().split()
 lineProgCount=lineProgCount+1
 if lineProgCount % 100000 ==0:
  print(lineProgCount)
 
 if len(vecarr) < 20:
  continue
 
 w2vecword = vecarr[0]
 if w2vecword in terms:
  termsVectors.append(vecline)

del vectorlines

termsVectorsDic = {}

for vecline in termsVectors:
 veclinearr = vecline.strip().split()
 vecword = veclinearr[0]
 vecnumbers = list(map(float, veclinearr[1:]))
 termsVectorsDic[vecword]=vecnumbers 

X_train = [] 
X_test = []
 
for i in range(len(train_textdata)): 
 words = train_textdata[i]
 sum_vecs = [0] * 300
 #print(len(sum_vecs), i)
 for word in words:
  if word in termsVectorsDic:
   for j in range(len(sum_vecs)):
    sum_vecs[j]=sum_vecs[j]+termsVectorsDic[word][j]
 
 X_train.append(sum_vecs)

for i in range(len(test_textdata)): 
 words = test_textdata[i]
 sum_vecs = [0] * 300
 #print(len(sum_vecs), i)
 for word in words:
  if word in termsVectorsDic:
   for j in range(len(sum_vecs)):
    sum_vecs[j]=sum_vecs[j]+termsVectorsDic[word][j]
 
 X_test.append(sum_vecs) 
 
 
#vectorizer = TfidfVectorizer( max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
#vectorizer = TfidfVectorizer(max_df=0.15, min_df=1, stop_words=stopwords, use_idf=True, smooth_idf=True, norm='l2')
#X_train = vectorizer.fit_transform(train_data)
#X_test = vectorizer.transform(test_data)


#scaler = MinMaxScaler(feature_range=(0, 1))
#X_train = scaler.fit_transform(X_train.toarray())
#X_test = scaler.transform(X_test.toarray())

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train.toarray())
#X_test = scaler.transform(X_test.toarray())


#t0 = time()
#ch2 = SelectKBest(chi2, k='all')
#X_train = ch2.fit_transform(X_train, train_labels)
#X_test = ch2.transform(X_test)
#print("done in %fs" % (time() - t0))

#ftrIndexes = ch2.get_support(indices=True)
#print(ftrIndexes)
#feature_names = vectorizer.get_feature_names()
#selected_feature_names = [feature_names[i] for i in ftrIndexes]
#print(selected_feature_names)

#clf=linear_model.LinearRegression()
clf = LogisticRegression() #52
#clf = RidgeClassifier(tol=1e-1) #52
#clf = Perceptron(n_iter=100)
#clf=KNeighborsClassifier(n_neighbors=10)
#clf = LinearSVC(loss='l2', C=1000, dual=False, tol=1e-3)
#clf = SGDClassifier(alpha=.0001, n_iter=50, penalty='l1')
#clf = SGDClassifier(alpha=.0001, n_iter=100, penalty='elasticnet')
#clf= MultinomialNB(alpha=.01)
#clf = BernoulliNB(alpha=.01)

t0 = time()
clf.fit(X_train, train_labels)
#clf.transform(X_test)
train_time = time() - t0
print ("train time: %0.3fs" % train_time)

t0 = time()
pred = clf.predict(X_test)
test_time = time() - t0
print ("test time:  %0.3fs" % test_time)

y_test = [int(i) for i in test_labels]
pred_test = [int(i) for i in pred]
score = metrics.homogeneity_score(y_test, pred_test)
print ("homogeneity_score:   %0.3f" % score)
score = metrics.completeness_score(y_test, pred_test)
print ("completeness_score:   %0.3f" % score)
score = metrics.v_measure_score(y_test, pred_test)
print ("v_measure_score:   %0.3f" % score)
score = metrics.accuracy_score(y_test, pred_test)
print ("acc_score:   %0.3f" % score)
score = metrics.normalized_mutual_info_score(y_test, pred_test)  
print ("nmi_score:   %0.3f" % score)

#file=open("D:/PhD/dr.norbert/dataset/shorttext/biomedical/semisupervised/biomedicalraw_ensembele_traintest","w")
file=open("/home/owner/PhD/dr.norbert/dataset/shorttext/agnews/semisupervised/agnewsraw_ensembele_traintest","w")
#file=open("D:/PhD/dr.norbert/dataset/shorttext/stackoverflow/semisupervised/stackoverflowraw_ensembele_traintest","w")
#file=open("D:/PhD/dr.norbert/dataset/shorttext/data-web-snippets/semisupervised/data-web-snippetsraw_ensembele_traintest","w")

for i in range(len(train_labels)):
 file.write(train_labels[i]+"\t"+train_trueLabels[i]+"\t"+train_data[i]+"\n")

for i in range(len(test_labels)):
 file.write(pred[i]+"\t"+test_labels[i]+"\t"+test_data[i]+"\n")

file.close()
