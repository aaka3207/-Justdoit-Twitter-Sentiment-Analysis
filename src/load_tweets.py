import pandas as pd
import csv
from pathlib import Path
import os
import tweepy
import numpy
consumer_key = 'CANCQIhBR2I8bJJDmkuJeXIWD'
consumer_secret = 'PoX61xEYsCuQzeBw0eB6Bau1tnK6vW4S1B3Klfuvz6doAlZbpb'
access_token = '627469112-JDgxN0NlMNJ9x4gfTUDea5sorp62IVfsnAiwPoHT'
access_token_secret = '3KPs6zF4AuuFkhiQ402bHg3bW8DkUbbJD8IXQrfJx1ndo'
import glob


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def convertToCsv(inputFile,output):

#trainTextPath = Path(__file__).absolute().parent.joinpath('../dataset/training/twitter-2016train-A.txt')
#trainCsvPth = Path(__file__).absolute().parent.joinpath('../dataset/training/train.csv')

    frame = pd.DataFrame()

    with open(inputFile, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split('\t') for line in stripped if line)
        with open( output, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('id', 'sentiment'))
            writer.writerows(lines)
            frame = pd.read_csv(output)
    return frame


def concatCsv(fullPath):
     # use your path
    allFiles = glob.glob(fullPath + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    return frame