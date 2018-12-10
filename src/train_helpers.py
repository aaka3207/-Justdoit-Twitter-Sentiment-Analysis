from pandas_confusion import ConfusionMatrix
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



def display_null_accuracy(y_test):
    value_counts = pd.value_counts(y_test)
    null_accuracy = max(value_counts) / float(len(y_test))
    print (('null accuracy: %s' % '{:.2%}').format(null_accuracy))
    return null_accuracy

def display_accuracy_score(y_test, y_pred_class):
    score = accuracy_score(y_test, y_pred_class)
    print (('accuracy score: %s' % '{:.2%}').format(score))
    return score

def display_accuracy_difference(y_test, y_pred_class):
    null_accuracy = display_null_accuracy(y_test)
    accuracy_score = display_accuracy_score(y_test, y_pred_class)
    difference = accuracy_score - null_accuracy
    return null_accuracy, accuracy_score

def train_test_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    print(pipeline.best_params_)
    print(pipeline.best_estimator_)
    y_pred_class = pipeline.predict(X_test)
    unique_label = np.unique(y_test)
    matrix = ConfusionMatrix(y_test, y_pred_class,labels=['True Value','Predicted Value'])
    display_accuracy_difference(y_test, y_pred_class)
    print ('-' * 75 + '\nConfusion Matrix\n')
    print (matrix)
    print('f1_score',f1_score(y_test, y_pred_class, average="macro"))
    print('precision',precision_score(y_test, y_pred_class, average="macro"))
    print('recall',recall_score(y_test, y_pred_class, average="macro"))    
    print(classification_report(y_test,y_pred_class))
    
    return pipeline, matrix.to_dataframe(), y_pred_class
