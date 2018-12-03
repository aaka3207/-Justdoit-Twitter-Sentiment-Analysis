from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from joblib import dump, load
from textblob import TextBlob
import clean
from sklearn_helpers import train_test_and_evaluate
from pathlib import Path
import warnings
import nltk
import gensim
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import collections
from pprint import pprint
import emoji
import os
import string
import re
from time import time
import pandas as pd
import csv
import numpy as np
pd.set_option('display.max_colwidth', -1)
warnings.filterwarnings('ignore')
np.random.seed(37)


trainTextPath = Path(__file__).absolute().parent.joinpath(
    '../dataset/train-preprocessed.csv')
testTextPath = Path(__file__).absolute().parent.joinpath(
    '../dataset/test-preprocessed.csv')

training_data = pd.read_csv(trainTextPath)
testing_data = pd.read_csv(testTextPath)

X_train, y_train = training_data.text, training_data.sentiment
X_test, y_test = testing_data.text, testing_data.sentiment
print('total training records', len(training_data))
print('total testing records', len(y_test))
print('total positive tweets:', len(
    training_data[training_data.sentiment == 'positive']))
print('total negative tweets:', len(
    training_data[training_data.sentiment == 'negative']))
print('total neutral tweets:', len(
    training_data[training_data.sentiment == 'neutral']))


# Your milage may vary on these arguments
tokenizer = nltk.casual.TweetTokenizer()
cv_vect = CountVectorizer(tokenizer=tokenizer.tokenize, analyzer="word",ngram_range=(1,2),max_df=.75)
#classifierCv = LogisticRegressionCV(multi_class='multinomial',solver='lbfgs',cv=4,penalty='l2')

tuned_parameters = {}
sentiment_pipeline = Pipeline([
    ('vect', cv_vect),
    ('tfidf', TfidfTransformer(use_idf=True,norm='l2')),
    ('clf', LogisticRegression(multi_class='multinomial',
                               solver='lbfgs', penalty='l2', class_weight="balanced",max_iter=4000))
])

#warning: this model is CPU intensive and is designed to utilize all cores. 
clf = GridSearchCV(sentiment_pipeline, tuned_parameters, cv=4, refit=True,n_jobs=-1)
sentiment_pipeline, confusion_matrix, y_true = train_test_and_evaluate(
    clf, X_train, y_train, X_test, y_test)
results = pd.DataFrame({'predicted': y_true, 'actual': y_test})
results.to_csv(Path(__file__).absolute().parent.joinpath(
    '../output/training_results.csv'))
dump(sentiment_pipeline, Path(__file__).absolute().parent.joinpath('../dataset/model.joblib'))
