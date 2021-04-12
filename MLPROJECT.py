#Here we are simply importing the things we will be using in our Script

from __future__ import print_function
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import eli5
import re
from tqdm import *


#Preprocessing¶
#Here are two utility functions to clean data,
#and optionally use the phonetic algorithm to hash the data.

def cleaner(word):
  word = re.sub(r'\#\.', '', word)
  word = re.sub(r'\n', '', word)
  word = re.sub(r',', '', word)
  word = re.sub(r'\-', ' ', word)
  word = re.sub(r'\.', '', word)
  word = re.sub(r'\\', ' ', word)
  word = re.sub(r'\\x\.+', '', word)
  word = re.sub(r'\d', '', word)
  word = re.sub(r'^_.', '', word)
  word = re.sub(r'_', ' ', word)
  word = re.sub(r'^ ', '', word)
  word = re.sub(r' $', '', word)
  word = re.sub(r'\?', '', word)

  return word.lower()

def hashing(word):
  word = re.sub(r'ain$', r'ein', word)
  word = re.sub(r'ai', r'ae', word)
  word = re.sub(r'ay$', r'e', word)
  word = re.sub(r'ey$', r'e', word)
  word = re.sub(r'ie$', r'y', word)
  word = re.sub(r'^es', r'is', word)
  word = re.sub(r'a+', r'a', word)
  word = re.sub(r'j+', r'j', word)
  word = re.sub(r'd+', r'd', word)
  word = re.sub(r'u', r'o', word)
  word = re.sub(r'o+', r'o', word)
  word = re.sub(r'ee+', r'i', word)
  if not re.match(r'ar', word):
    word = re.sub(r'ar', r'r', word)
  word = re.sub(r'iy+', r'i', word)
  word = re.sub(r'ih+', r'eh', word)
  word = re.sub(r's+', r's', word)
  if re.search(r'[rst]y', 'word') and word[-1] != 'y':
    word = re.sub(r'y', r'i', word)
  if re.search(r'[bcdefghijklmnopqrtuvwxyz]i', word):
    word = re.sub(r'i$', r'y', word)
  if re.search(r'[acefghijlmnoqrstuvwxyz]h', word):
    word = re.sub(r'h', '', word)
  word = re.sub(r'k', r'q', word)
  return word

def array_cleaner(array):
  # X = array
  X = []
  for sentence in array:
    clean_sentence = ''
    words = str(sentence).split(' ')
    for word in words:
      clean_sentence = clean_sentence +' '+ cleaner(word)
    X.append(clean_sentence)
  return X


#Data
#Here we are reading the file containing data
data = pd.read_csv('Dataset/Roman Urdu DataSet.csv', encoding="ISO-8859-1", header=None)
data.head()

#We are training the data on all of the dataset.
numpy_array = data.values
X = numpy_array[:, 0]
# Clean X here
X_train = array_cleaner(X)
y_train = numpy_array[:, 1]


#Vectorizing¶
#And using TF-IDF as our vectorizing method. We are specifying the N-gram to be 3.
ngram = 3
vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, ngram), max_df=0.5)
X_train = vectorizer.fit_transform(X_train)



#Classification¶
#A utility function to help us train different classifier:

def benchmark(clf, name):
  print('_' * 80)
  print("Training: ")
  print(clf)
  clf.fit(X_train, y_train)
  return clf

clf = benchmark(SGDClassifier(alpha=.0001,penalty="elasticnet"), 'SGD-elasticnet')
#clf = benchmark(LinearSVC(penalty='l1', dual=False,tol=1e-3), 'liblinear L1')
print(eli5.format_as_text(eli5.explain_weights(clf, vec=vectorizer)))

print("Enter To Analyse Sentiment Of A Sentence")
a="yes"
while a=="yes":
    test_sentence = input()
    print(eli5.format_as_text(eli5.explain_prediction(clf, doc=test_sentence, vec=vectorizer)))
    print("Enter yes If You Want To Enter Another Sentence")
    a=input()
