import re

##read labels obtained from external clustering algorithm
def readClustLabel(fileName):
 print("readClustLabel->", fileName)
 file1=open(fileName,"r", encoding="utf8")
 lines = file1.readlines()
 file1.close()
 clustlabels = []
 for line in lines:
  line = line.strip()
  arr = re.split(",", line) 
  clustlabels = clustlabels + arr

 print("readClustLabel->","total labels from "+ fileName+" is "+ str(len(clustlabels)))
 return clustlabels
