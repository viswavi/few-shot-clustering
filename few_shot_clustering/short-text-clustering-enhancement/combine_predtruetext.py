import re
from nltk.corpus import stopwords
from txt_process_util import processTxtRemoveStopWordTokenized

def combinePredTrueText(clustlabels, dataFileTxtTrue):
 listtuple_pred_true_text=[]
 file1=open(dataFileTxtTrue,"r", encoding="utf8")
 lines = file1.readlines()
 file1.close()
 if len(lines) != len(clustlabels):
  print("combinePredTrueText->#lines miss match #clustlabels") 
  return [] 
 terms=[]
 #stopWs = set(stopwords.words('english'))
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

 for i in range(len(clustlabels)):
  line = lines[i].strip()
  #print(line)
  arr = re.split("\t", line)
  predLabel = clustlabels[i]
  trueLabel = arr[0]
  text = arr[1]
  tupPredTrueTxt = [predLabel, trueLabel, text]  
  filtered_sentence = processTxtRemoveStopWordTokenized(text, stopWs)
  listtuple_pred_true_text.append(tupPredTrueTxt)
  terms.extend(filtered_sentence)

 terms=list(set(terms))
 print("#listtuple_pred_true_text is "+str(len(listtuple_pred_true_text))+" #terms="+str(len(terms))) 
 return [listtuple_pred_true_text, terms]
 
