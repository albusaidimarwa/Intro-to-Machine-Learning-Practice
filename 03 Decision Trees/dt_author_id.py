#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectPercentile

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


#########################################################
print('Number of features in the data is: ', len(features_train[0]))

def classify(features_train, labels_train, min_samples):

    ### your code goes here--should return a trained decision tree classifer
    X = features_train
    Y = labels_train
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples)

    # train the classifier
    t0 = time()
    clf = clf.fit(X,Y)
    print("Decision Tree Training time: " + str(round(time()-t0,3)) + " s")    
    return clf

clf_40 = classify(features_train, labels_train, 40)

# predict
t0 = time()
labels_pred_40 = clf_40.predict(features_test)
print("Decision Tree Prediction time: " + str(round(time()-t0,3)) + " s")

# accuracy
acc_min_samples_split_40 = accuracy_score(labels_test, labels_pred_40)

print("Decision Tree Predicted labels: " + str(len(labels_pred_40)))
print("Decision Tree Accuracy: " + str(accuracy_score(labels_test, labels_pred_40)))

