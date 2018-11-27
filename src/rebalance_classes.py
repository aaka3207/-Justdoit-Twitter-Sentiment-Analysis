from sklearn.utils import resample
from pathlib import Path
import pandas as  pd


extraClassPath =  Path(__file__).absolute().parent.joinpath('../dataset/train_from_test.csv')
extra_classes = pd.read_csv(extraClassPath,sep='\t')
extra_classes = extra_classes[['id','text','sentiment']]
negative_classes = extra_classes[extra_classes.sentiment == "negative"]
print(len(negative_classes))

negative_classes_upsampled = resample(negative_classes,replace=False,n_samples=2000,random_state=123)
print(negative_classes_upsampled)



training_data = pd.read_csv(Path(__file__).absolute().parent.joinpath('../dataset/train_full.csv'))
training_data = training_data[['id','text','sentiment']]

full_dataset = pd.concat([training_data,negative_classes_upsampled])
print(len(full_dataset))

full_dataset.to_csv(Path(__file__).absolute().parent.joinpath('../dataset/train_full.csv'))

