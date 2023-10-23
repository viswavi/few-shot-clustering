from sklearn.feature_extraction.text import TfidfVectorizer

def generateTrainTestVectorsTfIDf(trainTups_pred_true_txt, testTups_pred_true_txt):
 train_labels=[]
 test_labels=[]

 train_txt_datas= []
 for trainTup_pred_true_txt in trainTups_pred_true_txt:
  train_txt_datas.append(trainTup_pred_true_txt[2])
  train_labels.append(trainTup_pred_true_txt[0])

 test_txt_datas= []
 for testTup_pred_true_txt in testTups_pred_true_txt:
  test_txt_datas.append(testTup_pred_true_txt[2])
  test_labels.append(testTup_pred_true_txt[1])

 vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
 X_vec_train = vectorizer.fit_transform(train_txt_datas)
 X_vec_test = vectorizer.transform(test_txt_datas)

 return [X_vec_train, train_labels, X_vec_test, test_labels]
