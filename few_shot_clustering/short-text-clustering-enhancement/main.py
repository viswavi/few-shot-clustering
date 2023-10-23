from sklearn import metrics
from read_clust_label import readClustLabel
from combine_predtruetext import combinePredTrueText
from groupTxt_ByClass import groupTxtByClass
from nltk.tokenize import word_tokenize
from sent_vecgenerator import generate_sent_vecs_toktextdata
from generate_TrainTestTxtsTfIdf import comPrehensive_GenerateTrainTestTxtsByOutliersTfIDf
from generate_TrainTestVectorsTfIdf import generateTrainTestVectorsTfIDf
from sklearn.linear_model import LogisticRegression
from time import time
from nltk.corpus import stopwords
from txt_process_util import processTxtRemoveStopWordTokenized
import re
import numpy as np
import random
import sys
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from compute_util import MultiplyTwoSetsOneToOne

minIntVal = -1000000
numberOfClusters =8 #mandatory input
maxIterations=50
maxTrainRatio =0.85
#minTrainRatio = 0.60
minPercent = 70
maxPercent = 85
percentIncr = 5
#extParam = str(sys.argv[1])
#print(extParam)
'''trainFile = "/home/owner/PhD/dr.norbert/dataset/shorttext/googlenews/train"
testFile = "/home/owner/PhD/dr.norbert/dataset/shorttext/googlenews/test"
traintestFile = "/home/owner/PhD/dr.norbert/dataset/shorttext/googlenews/traintest"
textsperlabelDir="/home/owner/PhD/dr.norbert/dataset/shorttext/googlenews/semisupervised/textsperlabel/"
dataFileTrueTxt = "/home/owner/PhD/dr.norbert/dataset/shorttext/googlenews/S-original-order"
extClustFile = "/home/owner/PhD/clustering-k-means--/googlenews-11109-s-kmeans---glove-labels-"+extParam'''

trainFile = "data/search_snippets/train"
testFile = "data/search_snippets/test"
traintestFile = "data/search_snippets/traintest"
textsperlabelDir="data/search_snippets/semisupervised/textsperlabel/"
dataFileTrueTxt = "data/search_snippets/search_snippets_true_text"
extClustFile = "data/search_snippets/search_snippets_pred"

'''trainFile = "data/biomedical/train"
testFile = "data/biomedical/test"
traintestFile = "data/biomedical/traintest"
textsperlabelDir="data/biomedical/semisupervised/textsperlabel/"
dataFileTrueTxt = "data/biomedical/biomedical_true_text"
extClustFile = "data/biomedical/biomedical_pred"'''

'''trainFile = "data/stackoverflow/train" #output file: will be created by program
testFile = "data/stackoverflow/test" #output file: will be created by program
traintestFile = "data/stackoverflow/traintest" #output file: will be created by program
textsperlabelDir="data/stackoverflow/semisupervised/textsperlabel/" #please create this directory manually as we do not create by program because of permission issue
dataFileTrueTxt = "data/stackoverflow/stackoverflow_true_text" #input file: 
#data format
#18	How do you page a collection with LINQ?
#3	Best Subversion clients for Windows Vista (64bit)
extClustFile = "data/stackoverflow/stackoverflow_pred" #input file: this is the clustering output of a clustering algorithm
#data format
#3
#3'''


def WriteTrainTest(listtuple_pred_true_text, outFileName):
 file2=open(outFileName,"w", encoding="utf8")
 for i in range(len(listtuple_pred_true_text)):
  file2.write(listtuple_pred_true_text[i][0]+"\t"+listtuple_pred_true_text[i][1]+"\t"+listtuple_pred_true_text[i][2]+"\n")

 file2.close()


def ReadPredTrueText(InFileName):
 file1=open(InFileName,"r", encoding="utf8")
 lines = file1.readlines()
 file1.close()
 listtuple_pred_true_text = []
 for line in lines:
  line = line.strip()
  arr = re.split("\t", line)
  predLabel = arr[0]
  trueLabel = arr[1]
  text = arr[2]
  tupPredTrueTxt = [predLabel, trueLabel, text]
  listtuple_pred_true_text.append(tupPredTrueTxt) 
 
 return listtuple_pred_true_text


def MergeAndWriteTrainTest():
 print("MergeAndWriteTrainTest->",extClustFile)
 clustlabels=readClustLabel(extClustFile)
 listtuple_pred_true_text, uniqueTerms=combinePredTrueText(clustlabels, dataFileTrueTxt)
 WriteTrainTestInstances(traintestFile, listtuple_pred_true_text)
 return listtuple_pred_true_text 


def WriteTextsOfEachGroup(labelDir, dic_tupple_class):
 for label, value in dic_tupple_class.items():
  labelFile = labelDir+label
  file1=open(labelFile,"w", encoding="utf8")
  for pred_true_txt in value:
   file1.write(pred_true_txt[0]+"\t"+pred_true_txt[1]+"\t"+pred_true_txt[2]+"\n")

  file1.close()

def Gen_WriteOutliersEachGroup(labelDir, numberOfClusters):
 dic_label_outliers = {}
 for labelID in range(numberOfClusters):
  fileId = labelID#  +1 
  labelFile = labelDir+str(fileId)
  file1=open(labelFile,"r", encoding="utf8")
  lines = file1.readlines()
  file1.close()
  
  train_data = []
  train_labels = []
  train_trueLabels = []

  for line in lines:
   line=line.lower().strip() 
   arr = re.split("\t", line)
   train_data.append(arr[2])
   train_labels.append(arr[0])
   train_trueLabels.append(arr[1])

  vectorizer = TfidfVectorizer( max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
  x_train = vectorizer.fit_transform(train_data)

  contratio = 0.1
  isf = IsolationForest(n_estimators=100, max_samples='auto', contamination=contratio, max_features=1.0, bootstrap=True, verbose=0, random_state=0, behaviour="new")
  outlierPreds = isf.fit(x_train).predict(x_train)
  dic_label_outliers[str(fileId)] = outlierPreds  #real
  
  #dense_x_train = x_train.toarray()
  #outlierPreds_sd = detect_outlier_sd_vec(dense_x_train, 0.1)
  #outlierPredsMult = MultiplyTwoSetsOneToOne(outlierPreds, outlierPreds_sd)
  #outlierPreds=outlierPreds_sd
  #dic_label_outliers[str(fileId)] = outlierPreds #outlierPreds_sd #outlierPredsMult

  file1=open(labelDir+str(fileId)+"_outlierpred","w", encoding="utf8")
  for pred in outlierPreds:
   file1.write(str(pred)+"\n") 
 
  file1.close()
 
 return dic_label_outliers
 


def WriteTrainTestInstances(instFile, tup_pred_true_txts):
 file1=open(instFile,"w", encoding="utf8")
 for tup_pred_true_txt in tup_pred_true_txts:
  file1.write(tup_pred_true_txt[0]+"\t"+tup_pred_true_txt[1]+"\t"+tup_pred_true_txt[2]+"\n")  

 file1.close()



def GenerateTrainTest2_Percentage(percentTrainData):
 trainDataRatio = 1.0
		
 listtuple_pred_true_text = ReadPredTrueText(traintestFile)
 perct_tdata = percentTrainData/100
 goodAmount_txts = int(perct_tdata*(len(listtuple_pred_true_text)/numberOfClusters))			
 dic_tupple_class=groupTxtByClass(listtuple_pred_true_text, False)		
 #write texts of each group in  
 WriteTextsOfEachGroup(textsperlabelDir,dic_tupple_class)
 dic_label_outliers = Gen_WriteOutliersEachGroup(textsperlabelDir, numberOfClusters)

 train_pred_true_txts = []
 test_pred_true_txts = []

 for label, pred_true_txt in dic_tupple_class.items():
  outlierpreds = dic_label_outliers[str(label)]
  pred_true_txts = dic_tupple_class[str(label)]

  if len(outlierpreds)!= len(pred_true_txts):
   print("Size not match for="+str(label))
  
  outLiers_pred_true_txt = []
  count = -1
  for outPred in outlierpreds:
   outPred = str(outPred)
   count=count+1
   if outPred=="-1":
    outLiers_pred_true_txt.append(pred_true_txts[count])

  test_pred_true_txts.extend(outLiers_pred_true_txt)
  #remove outlierts insts from pred_true_txts
  pred_true_txts_good = [e for e in pred_true_txts if e not in outLiers_pred_true_txt]
  dic_tupple_class[str(label)]=pred_true_txts_good

  
 for label, pred_true_txt in dic_tupple_class.items():
  pred_true_txts = dic_tupple_class[str(label)] 
  pred_true_txt_subs= []
  numTrainGoodTexts=int(perct_tdata*len(pred_true_txts))
  if len(pred_true_txts) > goodAmount_txts:
   pred_true_txt_subs.extend(pred_true_txts[0:goodAmount_txts])
   test_pred_true_txts.extend(pred_true_txts[goodAmount_txts:len(pred_true_txts)]) 
  else:
   pred_true_txt_subs.extend(pred_true_txts)
  train_pred_true_txts.extend(pred_true_txt_subs)
 
 trainDataRatio = len(train_pred_true_txts)/len(train_pred_true_txts+test_pred_true_txts)
 #print("trainDataRatio="+str(trainDataRatio))
 if trainDataRatio<=maxTrainRatio:
  WriteTrainTestInstances(trainFile,train_pred_true_txts)
  WriteTrainTestInstances(testFile,test_pred_true_txts) 
   		
 return trainDataRatio
 


def PerformClassification(trainFile, testFile, traintestFile):
 file=open(trainFile,"r", encoding="utf8")
 lines = file.readlines()
 #np.random.seed(0)
 np.random.shuffle(lines)
 file.close()

 train_data = []
 train_labels = []
 train_trueLabels = []

 for line in lines:
  line=line.strip().lower() 
  arr = re.split("\t", line)
  train_data.append(arr[2])
  train_labels.append(arr[0]) #train_labels.append(arr[0])
  train_trueLabels.append(arr[1])
 
 file=open(testFile,"r", encoding="utf8")
 lines = file.readlines()
 file.close()

 test_data = []
 test_labels = []

 for line in lines:
  line=line.strip().lower() 
  arr = re.split("\t", line)
  test_data.append(arr[2])
  test_labels.append(arr[1])

 vectorizer = TfidfVectorizer( max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
 X_train = vectorizer.fit_transform(train_data)
 X_test = vectorizer.transform(test_data)
 
 clf = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=300) #52
 #t0 = time()
 clf.fit(X_train, train_labels)
 #train_time = time() - t0
 #print ("train time: %0.3fs" % train_time)

 #t0 = time()
 pred = clf.predict(X_test)
 #test_time = time() - t0
 #print ("test time:  %0.3fs" % test_time)

 y_test = [int(i) for i in test_labels]
 pred_test = [int(i) for i in pred]
 #score = metrics.homogeneity_score(y_test, pred_test)
 #print ("homogeneity_score-test-data:   %0.4f" % score)
 #score = metrics.normalized_mutual_info_score(y_test, pred_test)  
 #print ("nmi_score-test-data:   %0.4f" % score) 
 
 file=open(traintestFile,"w", encoding="utf8")
 for i in range(len(train_labels)):
  file.write(train_labels[i]+"\t"+train_trueLabels[i]+"\t"+train_data[i]+"\n")

 for i in range(len(test_labels)):
  file.write(pred[i]+"\t"+test_labels[i]+"\t"+test_data[i]+"\n")
 
 file.close()


def ComputePurity(dic_tupple_class):
 totalItems=0
 maxGroupSizeSum =0
 for label, pred_true_txts in dic_tupple_class.items():
  totalItems=totalItems+len(pred_true_txts)
  dic_tupple_class_originalLabel=groupTxtByClass(pred_true_txts, True)
  maxMemInGroupSize=minIntVal
  maxMemOriginalLabel=""
  for orgLabel, org_pred_true_txts in dic_tupple_class_originalLabel.items():
   if maxMemInGroupSize < len(org_pred_true_txts):
    maxMemInGroupSize=len(org_pred_true_txts)
    maxMemOriginalLabel=orgLabel
  
  maxGroupSizeSum=maxGroupSizeSum+maxMemInGroupSize
  
 acc=maxGroupSizeSum/totalItems
 #print("acc whole data="+str(acc))
 return acc

 
def EvaluateByPurity(traintestFile):
 listtuple_pred_true_text = ReadPredTrueText(traintestFile)
 preds = []
 trues = []
 for pred_true_text in listtuple_pred_true_text:
  preds.append(pred_true_text[0])
  trues.append(pred_true_text[1])
 
 #score = metrics.homogeneity_score(trues, preds)  
 #print ("homogeneity_score-whole-data:   %0.4f" % score)   			
 score = metrics.normalized_mutual_info_score(trues, preds, average_method='arithmetic')  
 #print ("nmi_score-whole-data:   %0.6f" % score)
 dic_tupple_class=groupTxtByClass(listtuple_pred_true_text, False)
 acc=ComputePurity(dic_tupple_class)
 print("acc", acc, "nmi", score)   
 


def GenerateTrainTest2List(listtuple_pred_true_text):
 print("---before iterative classification---")
 EvaluateByPurity(traintestFile)

 prevPercent=minPercent
 for itr in range(maxIterations):
  randPercent=random.randint(minPercent,maxPercent)
  absPercentDiff = abs(randPercent-prevPercent)
  if absPercentDiff<percentIncr:
   if randPercent >= prevPercent:
    randPercent = min(randPercent+percentIncr, maxPercent)
   elif randPercent < prevPercent:
    randPercent = max(randPercent-percentIncr, minPercent)
  prevPercent=randPercent
  trainDataRatio = GenerateTrainTest2_Percentage(randPercent);
  #print(str(itr)+","+str(randPercent))
  PerformClassification(trainFile, testFile, traintestFile)
  if itr==maxIterations-1:  
    print("---after iterative classification---")    
    EvaluateByPurity(traintestFile)
   
				
listtuple_pred_true_text = MergeAndWriteTrainTest()
GenerateTrainTest2List(listtuple_pred_true_text)


