#HILL-and-VALLEY-PREDICTION

#Objective: The goal of this project is to create a machine learning model using the Logistic Regression method to classify geographical locations as either hills or valleys. We'll train the model on a dataset that contains various geographical features (like elevation, slope, etc.) and labels that tell us whether each location is a hill or a valley. After training, we'll test the model on a different dataset to see how well it performs. This model could be useful in fields like geology, agriculture, and urban planning by helping to quickly identify hills and valleys in different areas.

#Data source: Kaggle.com

#IMPORT LIBRARIES
import pandas as pd
import numpy as np

#IMPORT DATASET
df = pd.read_csv('/content/Hill Valley Dataset.csv')

#DESCRIBE DATA
df.info()

#Define Y and X variable
y = df['Class']
X = df.drop('Class',axis=1)

#TRAIN TEST-SPLIT
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

#MODEL TRAIN
from sklearn.linear_model import LogisticRegression
lr.fit(X_train,y_train)

#MODEL PREDICTION
y_pred = lr.predict(X_test)
y_pred

#MODEL EVALUATION
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
