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
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)
from pathlib import Path
import os
import text_counts
import clean
from textblob import TextBlob
import columnExtractor
trainTextPath = Path(__file__).absolute().parent.joinpath('../dataset/train_full.csv')

training_data = pd.read_csv(trainTextPath,sep='\t')

training_data = training_data[['text','sentiment']]
training_data = training_data.dropna()
training_data = training_data.reindex(np.random.permutation(training_data.index))
text = training_data[training_data.text != 'Not Available']
training_data = text

positive_count = training_data['sentiment'].value_counts()
print(len(training_data), 'total records')

print('total positive tweets:', positive_count)

tc = text_counts.TextCounts()
df_eda = tc.fit_transform(training_data.text)
df_eda['sentiment'] = training_data.sentiment


ct = clean.CleanText()
sr_clean = ct.fit_transform(training_data.text)

sr_clean.sample(5)
print(sr_clean)
cv = CountVectorizer()
bow = cv.fit_transform(sr_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.show()
bag = pd.DataFrame([bow])
bag.to_csv(Path(__file__).absolute().parent.joinpath('../output/bow.csv'))



df_model = df_eda

df_model.columns.tolist()
print(df_model)
df_model.to_csv(Path(__file__).absolute().parent.joinpath('../output/train_output.csv'))


