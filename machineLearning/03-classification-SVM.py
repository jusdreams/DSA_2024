#This is from IBM ML0101EN Machine Learning with Python.
# Note to Add modules to github portfolio


#Module 3 - Classification
#Classification Types
#Decision Trees, Naïve Bayes, Linear Discriminant Analysis, K-nearest neighbor,Logistic regression, Neural Networks, and Support Vector Machines.

#Support Vector Machines
'''
Basically, SVMs are based on the idea of finding a hyperplane that best divides a dataset into
two classes, as shown here.
As were in a 2-dimensional space, you can think of the hyperplane as a line that linearly
separates the blue points from the red points.
One reasonable choice as the best hyperplane is the one that represents the largest separation,
or margin, between the two classes.
So, the goal is to choose a hyperplane with as big a margin as possible.
Examples closest to the hyperplane are support vectors.
It is intuitive that only support vectors matter for achieving our goal; and thus, other
training examples can be ignored.

applications - image recognition, text category assignment, and  detecting spam
'''
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, log_loss, f1_score
from sklearn.linear_model import LogisticRegression
import itertools
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('inline')

#load data
cell_df = pd.read_csv('/Users/jusdreams/Documents/GitHub/2024/DSA/data/cell_samples.csv')
cell_df.head()

'''
The ID field contains the patient identifiers. The characteristics of the cell samples from each patient are contained in fields Clump to Mit. The values are graded from 1 to 10, with 1 being the closest to benign.

The Class field contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are benign (value = 2) or malignant (value = 4).

Let's look at the distribution of the classes based on Clump thickness and Uniformity of cell size:
'''

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

#Preprocessing and selection
cell_df.dtypes

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()] #BareNuc has some non numerical values that can be dropped.
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

#We want the model to predict the value of Class (that is, benign (=2) or malignant (=4)). 
#As this field can have one of only two possible values, we need to change its measurement level to reflect this.

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y [0:5]

# Train/Test dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Modeling with SVM
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

#predict new values
yhat = clf.predict(X_test)
yhat [0:5]

# Evaluate

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

# Evaluate - Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Evaluate - Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

#F1 score and jaccard score
f1_score(y_test, yhat, average='weighted') 
jaccard_score(y_test, yhat, pos_label=2) 

#Practice building with linear kernel
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train) 
yhat2 = clf2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))