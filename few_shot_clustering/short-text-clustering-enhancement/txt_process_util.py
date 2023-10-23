import re
from nltk.tokenize import word_tokenize

def preprocess(str):
 str=re.sub(r'[^a-zA-Z0-9 ]', ' ', str)
 str=re.sub(r'\b[a-z]\b|\b\d+\b', '', str)
 str=re.sub(r'lt', ' ', str)
 str=re.sub(r'gt', ' ', str)
 str=re.sub(r'\s+',' ',str).strip()
 return str

def processTxtRemoveStopWordTokenized(txt, stopWs):
 str = preprocess(txt)
 word_tokens = word_tokenize(str)
 filtered_sentence = [w for w in word_tokens if not w in stopWs]
 return filtered_sentence
