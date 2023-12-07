#This is from IBM ML0101EN Machine Learning with Python.
# Note to Add modules to github portfolio

'''
Syllabus
Module 1: Introduction to Machine Learning
In this module, you will be introduced to various real world applications of machine learning, such as those in the healthcare, telecommunications and financial industries. You will also learn about the Python libraries that can be used to implement machine learning models and be introduced to the supervised and unsupervised algorithms.

Module 2: Regression
In this module, you will be introduced to simple linear, multiple linear and non-linear regression and their applications. You will also learn about model evaluation metrics for regression such as mean absolute error and mean squared error, as well as learning about two types of evaluation approaches: training and testing on the same dataset and train/test split. You will also have the opportunity to apply these methods in hands-on labs.

Module 3: Classification
In this module, you'll learn about classification algorithms such as K-Nearest Neighbors, Decision Trees, Logistic Regression and Support Vector Machine. You will also learn about various evaluation metrics for classifiers such as the Jaccard index, F1-score and Log Loss.

Module 4: Clustering
In this module, you will learn about clustering approaches such as K-Means Clustering, Hierarchical Clustering and Density-Based Spatial Clustering of Applications with Noise (DBSCAN). Clustering can be used for customer segmentation or grouping weather stations.Jaccard index, F1-score and Log Loss.

Module 5: Recommender Systems
In this week, you will learn about recommender systems. You will learn about the two main types of recommender systems: content-based and collaborative filtering, and you will have the opportunity to create these systems in the hands-on labs.
'''
#Module 1 - Intro to ML

#preprocessing - standarizing datasets
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X) 

#Train,Test, and Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#set up classification algorithim
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(X_train, y_train) #claasify unknown cases
clf.predict(X_test)

#evaluate model accuracy
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, yhat, labels=[1,0]))

#save model
import pickle
s = pickle.dumps(clf)

#we teach models by training it with labeled datasets
#supervised vs unsupervised learning (less controlled)
#Supervised->classical(discrete) or regression(continuous)
#unsupervised algorithims trains on the dataset without known outcomes or data used. (unlabled, clustering)
