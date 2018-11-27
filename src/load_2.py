import pandas as pd
import csv
from pathlib import Path
import glob
from load_tweets import convertToCsv,concatCsv
trainTextPath = Path(__file__).absolute().parent.joinpath('../dataset/2017_English_final/2017_English_final/DOWNLOAD/Subtask_A/train/twitter-2016train-A.txt')
trainOutPath = Path(__file__).absolute().parent.joinpath('../dataset/2017_English_final/2017_English_final/DOWNLOAD/Subtask_A/train/twitter-2016train-A.csv')

fullPath = "C:/Users/Sir/OneDrive/School/UIC/IDS 472/Final Project/main/dataset/2017_English_final/2017_English_final/DOWNLOAD/Subtask_A"

df = concatCsv(fullPath)
print(len(df))