from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import math

def compute_sim_matrix(txtVecs):
 rows= len(txtVecs)
 sim_matrix = [[0 for x in range(rows)] for y in range(rows)] 
 
 for i in range(rows):
  for j in range(i+1,rows,1):  
   sim_matrix[i][j]=compute_sim_value(txtVecs[i], txtVecs[j])  
   sim_matrix[j][i]=sim_matrix[i][j]

 return sim_matrix

def compute_sim_value(vecarr1, vecarr2):
 sim_value = 1- cosine(vecarr1, vecarr2) #cosine=distance
 return sim_value


def compute_mean_sd(numbers):
 meanVal = 0
 sdVal = 0
 sumVal = 0
 for num in numbers:
  sumVal = sumVal + num
 
 meanVal = sumVal/len(numbers)
 varainceSumVal = 0
 for num in numbers:
  varainceSumVal = varainceSumVal + (num-meanVal)*(num-meanVal)
 
 sdVal = math.sqrt(varainceSumVal/len(numbers))

 return [meanVal, sdVal]
 
def MultiplyTwoSetsOneToOne(set1, set2):
 if len(set1)!=len(set2):
  print("len_set1="+len(set1)+",len_set2="+len(set2))
  return set1

 merged = []
 for i in range(len(set1)):
  s1 = set1[i]
  s2 = set2[i]
  merged.append(s1*s2)

 return merged


def compute_row_sim_I(txtVec, txtVecs):
 rowSimsToI=[]
 for i in range(len(txtVecs)):
  simVal=compute_sim_value(txtVecs[i], txtVec)
  rowSimsToI.append(simVal)
  
 return rowSimsToI



  
 

