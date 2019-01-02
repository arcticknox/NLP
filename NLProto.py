#NLP Preprocessing

#Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Importing the dataset

#delimiter is tabs, and we ignore the double quotes in the dataset, in pandas quoting - 3 does that job

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#cleaning the text

import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 1000):

    # Everything other than a-z and A-Z (^) 
    review = re.sub('[^a-zA-Z]', ' ',  dataset['Review'][i]) 
    # Convert to lowercase
    review = review.lower()
    # Split the string so that comparison with each word is possible with the stopwords
    review = review.split()
    
    # Create an object of PorterStemmer class
    ps = PorterStemmer()
    
    #for loop for eliminating stopwords and also stemming is done
    review = [ ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    # Join the words back together as a single string
    review = ' '.join(review)
    
    corpus.append(review)
    
 # Creating the Bag Of Words Model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values   

#Using Classification model 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

     




