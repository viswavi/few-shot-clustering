def generate_sent_vecslist_toktextdatas(list_toktextdatas, termsVectorsDic, dim):
 print("Start generating sentence vecs list...")
 print(len(list_toktextdatas))
 
 list_toktextdatavecs = []

 for toktextdatas in list_toktextdatas:
  toktextdatavecs = generate_sent_vecs_toktextdata(toktextdatas, termsVectorsDic, dim)
  list_toktextdatavecs.append(toktextdatavecs)
    
 return list_toktextdatavecs


def generate_sent_vecs_toktextdata(toktextdatas, termsVectorsDic, dim):
 print("Start generating sentence vecs single...")
 print(len(toktextdatas))
 
 toktextdatavecs = []

 for i in range(len(toktextdatas)): 
  words = toktextdatas[i]
  sum_vecs = [0] * dim
  for word in words:
   if word in termsVectorsDic:
    for j in range(len(sum_vecs)):
     sum_vecs[j]=sum_vecs[j]+termsVectorsDic[word][j]
 
  toktextdatavecs.append(sum_vecs)  
    
 return toktextdatavecs 
