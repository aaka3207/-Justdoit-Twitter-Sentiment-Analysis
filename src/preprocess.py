import pandas as pd
import numpy as np 
pd.set_option('display.max_colwidth', -1)
from time import time

import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)
from pathlib import Path


import clean

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
ct_train = clean.CleanText()
sr_clean_train = ct_train.fit_transform(training_data.text)
training_data.text = sr_clean_train
training_data.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/train-preprocessed.csv'))
print('<---------- Training Data Preprocessed ----------->')

testing_data = testing_data[['text','sentiment']]
testing_data = testing_data.dropna()
testing_data = testing_data.reindex(np.random.permutation(testing_data.index))
text_test = testing_data[testing_data.text != 'Not Available']
testing_data = text_test
ct_test = clean.CleanText()
sr_clean_test = ct_test.fit_transform(testing_data.text)
testing_data.text = sr_clean_test
testing_data.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/test-preprocessed.csv'))
print('<---------- Testing Data Preprocessed ----------->')



