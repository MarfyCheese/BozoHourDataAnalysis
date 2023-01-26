import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
dataframe = pd.read_csv("Advertisement_Transcripts_deduped_edited.csv")
#print(dataframe.Ad_copy[1])




vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataframe.Ad_copy)
classify = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None).fit(X, dataframe.Category)
#print(X.shape)

test_submissions = dataframe.Ad_copy
X_new = vectorizer.transform(test_submissions)

predict = classify.predict(X_new)
print(np.mean(predict == dataframe.Category))
#print(predict)
#print(metrics.classification_report(dataframe.Category, predict))
