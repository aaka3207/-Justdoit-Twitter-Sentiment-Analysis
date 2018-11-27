import pandas as pd
import csv
from pathlib import Path
import os
import tweepy
from load_tweets import convertToCsv,getTweetFromId

consumer_key = 'CANCQIhBR2I8bJJDmkuJeXIWD'
consumer_secret = 'PoX61xEYsCuQzeBw0eB6Bau1tnK6vW4S1B3Klfuvz6doAlZbpb'
access_token = '627469112-JDgxN0NlMNJ9x4gfTUDea5sorp62IVfsnAiwPoHT'
access_token_secret = '3KPs6zF4AuuFkhiQ402bHg3bW8DkUbbJD8IXQrfJx1ndo'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


trainTextPath = Path(__file__).absolute().parent.joinpath('../dataset/training/twitter-2016train-A.txt')
trainCsvPth = Path(__file__).absolute().parent.joinpath('../dataset/training/train.csv')

train_set = pd.read_csv(trainCsvPth)
train_set = train_set[['id','sentiment']]
train_set['text'] = train_set['id'].apply(lambda x: getTweetFromId(x))


print(train_set.head(10))

train_set.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/training/train_with_messages_2.csv'))






