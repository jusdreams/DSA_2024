#This is from IBM ML0101EN Machine Learning with Python.
# Note to Add modules to github portfolio


#Module 3 - Classification
#Classification Types
#Decision Trees, Naïve Bayes, Linear Discriminant Analysis, K-nearest neighbor,Logistic regression, Neural Networks, and Support Vector Machines.

#Decsiion Trees
'''
Given a dataset, we can split data using the avalible attributes. 
Some attributes could be better suited or more precitive than other attributes. Looking for the purest nodes.
Entropy is the amount of randomness in the node. (Uncertainty). Entropy is used to calculate homogenius. Low entropy, low uniform distrbution, high purity.
0 being the purest, and 1 being impure.

Information gain and Entropy can be considered as inverses or opposites. (Entropy before split -  Average Entropy After Split) (E1 - ((E2 + E3)/2) )
Higher information gain = more pure
'''
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#imports
import sys
import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

matplotlib.use('inline')

#load data
my_data = pd.read_csv("/Users/jusdreams/Documents/GitHub/2024/DSA/data/drug200.csv")

my_data.head() # take a look at the dataset

my_data.shape() #size of data

#preprocessing
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

y = my_data["Drug"] #Fill the target variable

#Set up decision tree
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
#check if x and y training sets match
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
#check if x and y test sets match
print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters

drugTree.fit(X_trainset,y_trainset) #fill with training set

predTree = drugTree.predict(X_testset)

#compare predictions to actual values
print (predTree [0:5])
print (y_testset [0:5])

#Evaluate
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
tree.plot_tree(drugTree)
plt.show()