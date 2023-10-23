from groupTxt_ByClass import groupTxtByClass
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import collections

def generateTrainTestTxtsByOutliers(dic_tuple_class, dic_list_outliers_class1, maxItemsInEachClass):
 trainTup_pred_true_txt= []
 testTup_pred_true_txt=[]

 for key, value in dic_tuple_class.items():
  if key not in dic_list_outliers_class1 or len(value) != len(dic_list_outliers_class1[key]):	  
   print("miss match="+key)
   continue
 
  rest_traintupTPTxt = []
  outliers= dic_list_outliers_class1[key]
  print("collections.Counter="+str(collections.Counter(outliers))+", key="+key+", len(outliers)="+str(len(outliers))+", len(value)="+str(len(value)))
  count=-1
  for tup_pred_true_text in value:
   count=count+1
   #print("outliers[count]="+str(outliers[count]))
   if outliers[count]==-1:
    testTup_pred_true_txt.append(tup_pred_true_text)
   else:
    rest_traintupTPTxt.append(tup_pred_true_text)
   
  print("len(rest_traintupTPTxt)="+str(len(rest_traintupTPTxt))+",maxItemsInEachClass="+str(maxItemsInEachClass))
  if len(rest_traintupTPTxt) > maxItemsInEachClass:
   trainTup_pred_true_txt.extend(rest_traintupTPTxt[0:maxItemsInEachClass])
   testTup_pred_true_txt.extend(rest_traintupTPTxt[maxItemsInEachClass:len(rest_traintupTPTxt)])
  else:
   trainTup_pred_true_txt.extend(rest_traintupTPTxt)

 
 print("after remove outlier, max items="+str(maxItemsInEachClass)+", total="+str(len(trainTup_pred_true_txt+testTup_pred_true_txt)))
 groupTxtByClass(trainTup_pred_true_txt+testTup_pred_true_txt, False) 
 return [trainTup_pred_true_txt, testTup_pred_true_txt]

def comPrehensive_GenerateTrainTestTxtsByOutliersTfIDf(listtuple_pred_true_text, maxItemsInEachClass):
 trainTup_pred_true_txt= []
 testTup_pred_true_txt=[]

 print("before remove outlier")
 dic_tuple_class = groupTxtByClass(listtuple_pred_true_text, False)

 contratio = 0.1
 dic_list_outliers_class = {}

 for key, value in dic_tuple_class.items():
  txt_datas= []
  for tup_pred_true_text in value:
   txt_datas.append(tup_pred_true_text[2])

  #print(len(txt_datas))
  vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
  x_train = vectorizer.fit_transform(txt_datas)
  isf = IsolationForest(n_estimators=100, max_samples='auto', contamination=contratio, max_features=1.0, bootstrap=True, verbose=0, random_state=0)
  outlierPreds = isf.fit(x_train).predict(x_train)
  #print(len(outlierPreds))
  dic_list_outliers_class[key]=outlierPreds

 trainTup_pred_true_txt, testTup_pred_true_txt = generateTrainTestTxtsByOutliers(dic_tuple_class, dic_list_outliers_class, maxItemsInEachClass)
 print("#trainTup_pred_true_txt="+str(len(trainTup_pred_true_txt))+", #testTup_pred_true_txt="+str(len(testTup_pred_true_txt)))   
 return [trainTup_pred_true_txt, testTup_pred_true_txt]
