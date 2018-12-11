from joblib import dump, load
from pathlib import Path
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
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
np.random.seed(37)

#load our model 
model = load(Path(__file__).absolute().parent.joinpath(
    "../dataset/model.joblib"))
print('------------model loaded----------')
#load our preproccessed Nike twitter data
testTextPath = Path(__file__).absolute().parent.joinpath(
    '../dataset/nike-preprocessed.csv')
print('---------dataset loaded-----------')
testing_data = pd.read_csv(testTextPath)
testing_data = testing_data[['text_clean',
                             'tweet_created_at', 'tweet_full_text']]
#we only want to use the preprocessed text for predicting
X_test = testing_data.text_clean
#predict the tweet sentiment
print('----------predicting...------------')
y_pred_class = model.predict(X_test)
#append sentiment to dataframe
testing_data['sentiment'] = y_pred_class
#append probability of correct prediction to dataframe
probability = model.predict_proba(X_test)
testing_data['probability'] = np.linalg.norm(probability, axis=1)
#remove preprocessed text from report.
del testing_data['text_clean']
testing_data.to_csv(Path(__file__).absolute().parent.joinpath(
    '../output/nike-sentiment-results.csv'))
print('-------report saved---------')
