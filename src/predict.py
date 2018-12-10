import pandas as pd
import csv
import numpy as np 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
import emoji
from pprint import pprint
import collections
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.externals import joblib
np.random.seed(37)
from pathlib import Path
import os
from joblib import dump, load


import clean
from textblob import TextBlob
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from pandas_confusion import ConfusionMatrix


model = load(Path(__file__).absolute().parent.joinpath("../dataset/model.joblib"))

testTextPath =  Path(__file__).absolute().parent.joinpath('../dataset/nike-preprocessed.csv')
testing_data = pd.read_csv(testTextPath)
testing_data = testing_data[['text_clean','tweet_created_at','tweet_full_text']]
X_test = testing_data.text_clean
print(X_test.head(10))

y_pred_class = model.predict(X_test)
testing_data['sentiment'] = y_pred_class
probability = model.predict_proba(X_test)
testing_data['probability'] = np.linalg.norm(probability,axis=1)
del testing_data['text_clean']
testing_data.to_csv(Path(__file__).absolute().parent.joinpath('../output/nike-sentiment-results.csv'))
print('report saved')