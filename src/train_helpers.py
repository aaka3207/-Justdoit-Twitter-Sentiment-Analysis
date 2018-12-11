from pandas_confusion import ConfusionMatrix
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix




def train_test_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred_class = pipeline.predict(X_test)
    unique_label = np.unique(y_test)
    matrix = ConfusionMatrix(y_test, y_pred_class,labels=['True Value','Predicted Value'])
    print ('-' * 75 + '\nConfusion Matrix\n')
    print (matrix)
    print('f1_score',f1_score(y_test, y_pred_class, average="macro"))
    print('precision',precision_score(y_test, y_pred_class, average="macro"))
    print('recall',recall_score(y_test, y_pred_class, average="macro"))    
    
    return pipeline, matrix.to_dataframe(), y_pred_class
