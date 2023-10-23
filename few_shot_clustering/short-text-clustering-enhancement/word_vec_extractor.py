def extract_word_vecs_list(list_toktextdatas, embeddingfile, dim):
 print("list_toktextdatas", len(list_toktextdatas))
 
 terms = [] 

 for toktextdatas in list_toktextdatas:
  for word_tokens in toktextdatas:
   terms.extend(word_tokens)

 terms=set(terms)
 print("terms length", len(terms))

 file=open(embeddingfile,"r")
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
  
 print("termsVectorsDic length", len(termsVectorsDic))
 
 return termsVectorsDic


def extract_word_vecs(toktextdatas, embeddingfile, dim):
 print("toktextdatas", len(toktextdatas))
 
 terms = [] 

 for word_tokens in toktextdatas:
  terms.extend(word_tokens)

 terms=set(terms)
 print("terms length", len(terms))

 file=open(embeddingfile,"r")
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
  
 print("termsVectorsDic length", len(termsVectorsDic))
 
 return termsVectorsDic


def populateTermVecs(terms, embeddingfile, dim):
 termsVectorsDic = {}

 file=open(embeddingfile,"r")
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
  
 print("termsVectorsDic length", len(termsVectorsDic))

 return termsVectorsDic
  
