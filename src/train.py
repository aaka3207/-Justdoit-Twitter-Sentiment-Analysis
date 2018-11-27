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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.externals import joblib
from sklearn.svm import SVC
import gensim
import nltk
import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)
from pathlib import Path
import os

from sklearn_helpers import train_test_and_evaluate

import clean
from textblob import TextBlob



trainTextPath = Path(__file__).absolute().parent.joinpath('../dataset/train-preprocessed.csv')
testTextPath =  Path(__file__).absolute().parent.joinpath('../dataset/test-preprocessed.csv')

training_data = pd.read_csv(trainTextPath)
testing_data = pd.read_csv(testTextPath)

X_train,y_train = training_data.text,training_data.sentiment
X_test,y_test = testing_data.text,testing_data.sentiment
print('total training records',len(training_data))
print('total testing records',len(y_test))
print('total positive tweets:',len(training_data[training_data.sentiment == 'positive']))
print('total negative tweets:',len(training_data[training_data.sentiment == 'negative']))
print('total neutral tweets:',len(training_data[training_data.sentiment == 'neutral']))


tokenizer = nltk.casual.TweetTokenizer() # Your milage may vary on these arguments
count_vect = TfidfVectorizer(tokenizer=tokenizer.tokenize) 
tfid_vect = TfidfVectorizer(tokenizer=tokenizer.tokenize) 
classifier = LogisticRegression(multi_class='multinomial',solver='lbfgs')
classifierCv = LogisticRegressionCV(multi_class='multinomial',solver='lbfgs',cv=4,penalty='l2')
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'cv':[3,4],'class_weight':['balanced'],'multi_class':['multinomial'] }
svm = SVC(kernel='linear', 
            class_weight='balanced', # penalize
            probability=True)
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
sentiment_pipeline = Pipeline([
        ('vectorizer', tfid_vect),
        ('classifier',classifierCv )
    ])

sentiment_pipeline, confusion_matrix,y_true = train_test_and_evaluate(sentiment_pipeline, X_train, y_train, X_test, y_test)
results = pd.DataFrame({'predicted':y_true,'actual':y_test})
results.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/results.csv'))