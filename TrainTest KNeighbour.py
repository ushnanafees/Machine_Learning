# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:55:56 2018

@author: Ushaan
"""
from sklearn import datasets      # Import a data set
iris = datasets.load_iris()

x = iris.data               # Input data (feature)
y = iris.target             # label 

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .6)

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
test = accuracy_score(y_test, predictions)
print(test) 
