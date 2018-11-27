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

nikeTweetPath = Path(__file__).absolute().parent.joinpath('../dataset/5000-justdoit-tweets-dataset/justdoit_tweets_2018_09_07_2.csv')
tweet_data = pd.read_csv(nikeTweetPath)


tweet_data = tweet_data[['tweet_created_at','tweet_full_text']]
tweet_data = tweet_data.dropna()
tweet_data = tweet_data.reindex(np.random.permutation(tweet_data.index))
#training_data['sentiment'] = training_data['sentiment'].map({'negative': -1, 'positive': 1,'neutral':.5})
ct = clean.CleanText()
sr_clean = ct.fit_transform(tweet_data.tweet_full_text)
print(sr_clean)
tweet_data['text_clean'] = sr_clean

tweet_data.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/nike-preprocessed.csv'))


def load_and_preprocess():
    nikeTweetPath = Path(__file__).absolute().parent.joinpath('../dataset/5000-justdoit-tweets-dataset/justdoit_tweets_2018_09_07_2.csv')
    tweet_data = pd.read_csv(nikeTweetPath)


    tweet_data = tweet_data[['tweet_created_at','tweet_full_text']]
    tweet_data = tweet_data.dropna()
    tweet_data = tweet_data.reindex(np.random.permutation(tweet_data.index))
    #training_data['sentiment'] = training_data['sentiment'].map({'negative': -1, 'positive': 1,'neutral':.5})
    ct = clean.CleanText()
    sr_clean = ct.fit_transform(tweet_data.tweet_full_text)
    print(sr_clean)
    tweet_data['text_clean'] = sr_clean

    tweet_data.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/nike-preprocessed.csv'))