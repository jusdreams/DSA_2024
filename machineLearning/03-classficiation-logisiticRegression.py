#This is from IBM ML0101EN Machine Learning with Python.
# Note to Add modules to github portfolio


#Module 3 - Classification
#Classification Types
#Decision Trees, Naïve Bayes, Linear Discriminant Analysis, K-nearest neighbor,Logistic regression, Neural Networks, and Support Vector Machines.

#Logisitic Regression
'''
use one or more independent variables(input) to create a dependent variable (output)
- difference between linear regression and logistic regression is that we are looking for a binary result (Yes or no) rather than a continous value (income)
'''
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, log_loss
from sklearn.linear_model import LogisticRegression
import itertools
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('inline')

#load data
chrun_df = pd.read_csv("/Users/jusdreams/Documents/GitHub/2024/DSA/data/ChurnData.csv")
chrun_df.head() # take a look at the dataset

#preprocessing
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

chrun_df.shape #how many rows and columns

#define X (independent variable)
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

#define y (dependent variable)
y = np.asarray(churn_df['churn'])
y [0:5]

#normalize data
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Model
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

#predict using test set
yhat = LR.predict(X_test)
yhat

#predict probability
yhat_prob = LR.predict_proba(X_test)
yhat_prob

#Evaluate - jaccard index
jaccard_score(y_test, yhat,pos_label=0)

#Evaluate - plot conusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Evaluate - Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Evaluate - Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))

'''
Based on the count of each section, we can calculate precision and recall of each label:

    Precision is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)

    Recall is the true positive rate. It is defined as: Recall =  TP / (TP + FN)

So, we can calculate the precision and recall of each class.

F1 score: Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label.

The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.

Finally, we can tell the average accuracy for this classifier is the average of the F1-score for both labels, which is 0.72 in our case.

'''
#Build Logistic Regression with same dataset but use differnt __solver__ and __regularization__ values.
LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))