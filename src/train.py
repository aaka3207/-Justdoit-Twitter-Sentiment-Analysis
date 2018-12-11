from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from joblib import dump, load
import clean
from train_helpers import train_test_and_evaluate
from pathlib import Path
import warnings
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
import collections
from pprint import pprint
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

#path of preprocessed training data
trainTextPath = Path(__file__).absolute().parent.joinpath(
    '../dataset/train-preprocessed.csv')
#path of preprocessed testing data

testTextPath = Path(__file__).absolute().parent.joinpath(
    '../dataset/test-preprocessed.csv')
#loading training data
training_data = pd.read_csv(trainTextPath)
#loading testing data
testing_data = pd.read_csv(testTextPath)
#for both training and testing data, we only want to use the text and sentiment columns
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


#we will be using nltk's TweetTokenizer to seperate every word in each tweet into an individual token
tokenizer = nltk.casual.TweetTokenizer()
#Scikit Learn's Countvectorizer, used to tranform our training and testing data into a bag of words model
cv_vect = CountVectorizer(tokenizer=tokenizer.tokenize, analyzer="word",ngram_range=(1,2),max_df=.75)

#if we want to use GridSearchCV to try out different parameters for the pipeline
tuned_parameters = {}
#our training pipeline. The pipeline will first transform our model 
sentiment_pipeline = Pipeline([
    ('vect', cv_vect),
    ('tfidf', TfidfTransformer(use_idf=True,norm='l2')),
    ('clf', LogisticRegression(multi_class='multinomial',
                               solver='lbfgs', penalty='l2', class_weight="balanced"))
])

#warning: this model is CPU intensive, may take a while. 
clf = GridSearchCV(sentiment_pipeline, tuned_parameters, cv=4, refit=True)
print('-----------Training Model-----------')
sentiment_pipeline, confusion_matrix, y_true = train_test_and_evaluate(
    clf, X_train, y_train, X_test, y_test)
results = pd.DataFrame({'predicted': y_true, 'actual': y_test})
results_path = '~/../output/training-results.xlsx'
writer = pd.ExcelWriter(results_path)
results.to_excel(writer,'Results')
print('--------Writing raw testing results to final results --------------')
confusion_matrix.to_excel(writer,'Confusion Matrix')
print('----------Writing confusion matrix to final results ----------------')
writer.save()
print('----------Final report saved----------------')
dump(sentiment_pipeline, Path(__file__).absolute().parent.joinpath('../dataset/model.joblib'))
print('------------Model saved to disk-------------')