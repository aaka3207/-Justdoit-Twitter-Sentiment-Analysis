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
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
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
import clean_2

trainTextPath = Path(__file__).absolute().parent.joinpath('../dataset/train_full.csv')
testTextPath =  Path(__file__).absolute().parent.joinpath('../dataset/test_full.csv')
training_data = pd.read_csv(trainTextPath)
testing_data = pd.read_csv(testTextPath,sep='\t')

print('<---------- Preprocessing Data ----------->')
training_data = training_data[['text','sentiment']]
training_data = training_data.dropna()
training_data = training_data.reindex(np.random.permutation(training_data.index))
text_train = training_data[training_data.text != 'Not Available']
training_data = text_train
ct_train = clean_2.CleanText()
sr_clean_train = ct_train.fit_transform(training_data.text)
training_data.text = sr_clean_train
training_data.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/train-preprocessed.csv'))
print('<---------- Training Data Preprocessed ----------->')

testing_data = testing_data[['text','sentiment']]
testing_data = testing_data.dropna()
testing_data = testing_data.reindex(np.random.permutation(testing_data.index))
text_test = testing_data[testing_data.text != 'Not Available']
testing_data = text_test
ct_test = clean_2.CleanText()
sr_clean_test = ct_test.fit_transform(testing_data.text)
testing_data.text = sr_clean_test
testing_data.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/test-preprocessed.csv'))
print('<---------- Testing Data Preprocessed ----------->')



