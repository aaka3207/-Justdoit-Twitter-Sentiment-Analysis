import pandas as pd
import numpy as np 
pd.set_option('display.max_colwidth', -1)


import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)
from pathlib import Path
import clean

nikeTweetPath = Path(__file__).absolute().parent.joinpath('../dataset/5000-justdoit-tweets-dataset/justdoit_tweets_2018_09_07_2.csv')
tweet_data = pd.read_csv(nikeTweetPath)


tweet_data = tweet_data[['tweet_created_at','tweet_full_text']]
tweet_data = tweet_data.dropna()
tweet_data = tweet_data.reindex(np.random.permutation(tweet_data.index))
#training_data['sentiment'] = training_data['sentiment'].map({'negative': -1, 'positive': 1,'neutral':.5})
ct = clean.CleanText()
sr_clean = ct.fit_transform(tweet_data.tweet_full_text)
tweet_data['text_clean'] = sr_clean

tweet_data.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/nike-preprocessed.csv'))


