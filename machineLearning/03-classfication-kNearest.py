#This is from IBM ML0101EN Machine Learning with Python.
# Note to Add modules to github portfolio


#Module 3 - Classification
#Classification Types
#Decision Trees, Naïve Bayes, Linear Discriminant Analysis, K-nearest neighbor,Logistic regression, Neural Networks, and Support Vector Machines.

#K-nearest neighbor
'''
Given a dataset, with predefined labels, we need to build a model to be used
to predict the class of a new or unknown case. Will use their similarities or dissimilarities to see what neighbors exist

1. Pick a value for K.
2. Calculate the distance from the new case (holdout from each of the cases in the dataset).
3. Search for the K observations in the training data that are ‘nearest’ to the measurements
of the unknown data point.
4. predict the response of the unknown data point using the most popular response value from
the K nearest neighbors.

Distances can be calculated in one or multi dimensional spaces using a distance matrix formula. Goal is to find best accuracy

Evaluation Metrics (Jaccard index, F-1 score, Log Loss)
Jaccard uses actual labels(y) and predicted labels (y^). index is the intersection of labels
F-1 Score, confusion matrix [actual, corrected wrong predictions] *identify errors, seperate classes [True Pos, False Neg, True Neg, False Pos]
    Precision = TP / (TP+FP)
    Recall = TP / (TP + FN)
    F-1 Score = 2x(precision x recall) / (precision + recall)
Log Loss measures performance of outlier. How far is the prediciton from the actual label. Lower log loss better accuracy

The target field, called custcat, has four possible values that correspond to the four customer groups, as follows: 1- Basic Service 2- E-Service 3- Plus Service 4- Total Service
'''
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import preprocessing

matplotlib.use('inline')

#load data
df = pd.read_csv("data/teleCust1000t.csv")

df.head() # take a look at the dataset

df['custcat'].value_counts() #how many of each class in dataset

df.hist(column='income', bins=50) #visualize using histogram

df.columns # define feature sets, X

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values  #.astype(float)
X[0:5]

y = df['custcat'].values #define our labels, y
y[0.5]

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float)) #Normalize data
X[0:5]

#Train test split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Classification
from sklearn.neighbors import KNeighborsClassifier

k = 4
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
neigh

#We can use this model to make predictions on the test set
yhat = neigh.predict(X_test)
yhat[0:5]

#Accuracy Eval
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))