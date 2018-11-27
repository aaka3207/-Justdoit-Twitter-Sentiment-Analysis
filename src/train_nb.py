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
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import GridSearchCV,KFold
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
from joblib import dump, load
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn_helpers import train_test_and_evaluate

import clean
from textblob import TextBlob



trainTextPath = Path(__file__).absolute().parent.joinpath('../dataset/train-preprocessed.csv')
testTextPath =  Path(__file__).absolute().parent.joinpath('../dataset/test-preprocessed.csv')

training_data = pd.read_csv(trainTextPath)
testing_data = pd.read_csv(testTextPath)

training_data = training_data[['text','sentiment']]
testing_data = testing_data[['text','sentiment']]

X_train,y_train = training_data.text,training_data.sentiment
X_test,y_test = testing_data.text,testing_data.sentiment
print('total training records',len(training_data))
print('total testing records',len(y_test))
negative_counts = training_data[training_data.sentiment == 'negative']
print('total positive tweets:',len(training_data[training_data.sentiment == 'positive']))
print('total negative tweets:',len(training_data[training_data.sentiment == 'negative']))
print('total neutral tweets:',len(training_data[training_data.sentiment == 'neutral']))

tokenizer = nltk.casual.TweetTokenizer(reduce_len=True) # Your milage may vary on these arguments
tfid_vect = TfidfVectorizer(tokenizer=tokenizer.tokenize,use_idf=False, ngram_range=(1, 2),max_df=.5,min_df=1)
cv_vect = CountVectorizer(tokenizer=tokenizer.tokenize)
classifierCv = LogisticRegressionCV(multi_class='multinomial',max_iter=100,solver='lbfgs',cv=KFold(n_splits=4,shuffle=True),penalty='l2',refit=True,Cs=[1],class_weight='balanced')
param_grid = {'Cs': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'cv':[3,4],'class_weight':['balanced'],'multi_class':['multinomial'] }

clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
sentiment_pipeline = Pipeline([

        ('vect', cv_vect),
        ('tfidf', TfidfTransformer()),
        ('clf',MultinomialNB() )
    ])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}
clf = GridSearchCV(sentiment_pipeline, tuned_parameters, cv=4,refit=True)

sentiment_pipeline, confusion_matrix,y_true = train_test_and_evaluate(clf, X_train, y_train, X_test, y_test)
dump(sentiment_pipeline,Path(__file__).absolute().parent.joinpath("../output/model.joblib"))
results = pd.DataFrame({'predicted':y_true,'actual':y_test})
results.to_csv(Path(__file__).absolute().parent.joinpath('../output/training-results.csv'))