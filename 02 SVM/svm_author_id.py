#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

from sklearn import svm

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

#kernel = "linear" or "rbf
#C = 1, 10, 100, 1000, 10000
clf = svm.SVC(kernel = "rbf", C=10000)

t0 = time()
clf.fit(features_train, labels_train)

#print training time
print('Training Time is: ', round(time()-t0, 3), "s")

t0 = time()
# Predicting time
clf.predict(features_test)
print('Predicting Time of is: ', round(time()-t0, 3), "s")

# Author ID Accuracy
Accuracy = clf.score(features_test, labels_test) 
print('Accuracy is: ', Accuracy)

pred=clf.predict(features_test)
print ("Prediction for element 10th, 26th and 50th are:", pred[10], pred[26], pred[50])

print('Number of events predicted in Chris class is', sum(clf.predict(features_test) ==1))

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
