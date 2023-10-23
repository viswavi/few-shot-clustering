def groupItemsBySingleKeyIndex(listItems, keyIndex):
  dic_itemGroups={}
  for item in listItems:
    key=str(item[keyIndex]) 
    dic_itemGroups.setdefault(key, []).append(item)
  
  return dic_itemGroups

##group txt by class
def groupTxtByClass(listtuple_pred_true_text, isByTrueLabel):
 dic_tupple_class = {}
 if isByTrueLabel == False:
  #print("isByPredLabel")
  for tuple_pred_true_text in listtuple_pred_true_text:
   predLabel = tuple_pred_true_text[0]
   trueLabel = tuple_pred_true_text[1]
   txt = tuple_pred_true_text[2]  
   dic_tupple_class.setdefault(predLabel, []).append([predLabel, trueLabel, txt])
 else:
  #print("isByTrueLabel")
  for tuple_pred_true_text in listtuple_pred_true_text:
   predLabel = tuple_pred_true_text[0]
   trueLabel = tuple_pred_true_text[1]
   txt = tuple_pred_true_text[2]  
   dic_tupple_class.setdefault(trueLabel, []).append([predLabel, trueLabel, txt])  

 #for key, value in dic_tupple_class.items():
 # print(key, len(value))

 return dic_tupple_class
