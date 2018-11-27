import glob
import pandas as pd

path = 'C:/Users/Sir/OneDrive/School/UIC/IDS 472/Final Project/main/dataset/2017_English_final/2017_English_final/GOLD/Subtask_A/train'
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
print(allFiles)
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    print(len(df))
    list_.append(df)
frame = pd.concat(list_, ignore_index=True)

frame.to_csv('../train/full.csv')