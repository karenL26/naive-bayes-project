########################################
# Naive Bayes Project Project          #
########################################
### Load libraries and modules ###
# Dataframes and matrices ----------------------------------------------
import pandas as pd
import numpy as numpy
import joblib
import os
# Machine learning -----------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
# Metrics --------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score

######################
# Data Preprocessing #
######################
# Loading the dataset
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')
# Create a copy of the original dataset
df = df_raw.copy()
# Remove package name as it's not relevant
df = df.drop('package_name', axis=1)
# Convert text to lowercase
df['review'] = df['review'].str.strip().str.lower()

#####################
# Model and results #
#####################
### It is implemented the model with the better results, tis information is in the explore notebook #

#Separate predictor from target
X = df['review']
y = df['polarity']
# Spliting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english')
X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print(y_pred)
#### Prediction ####
print(mnb.predict(vec.transform(['Love this app simply awesome!'])))
#### Model Evaluation ####
cm = confusion_matrix(y_test, y_pred, labels=mnb.classes_)
print("Confusion Matrix", cm)
print("\n")
print(classification_report(y_test,y_pred))
print("\n")
print("Multinomial Naive Bayes Mean absolute error:", mean_absolute_error(y_test, y_pred))
print("\n")
print('Multinomial Naive Bayes Train Accuracy = ',accuracy_score(y_train,mnb.predict(X_train)))
print('Multinomial Naive Bayes Test Accuracy = ',accuracy_score(y_test,mnb.predict(X_test)))
print("\n")
print("Multinomial Naive Bayes Precision score:",precision_score(y_test, y_pred))
print("\n")
print("Multinomial Naive Bayes Recall score:",recall_score(y_test, y_pred))

####################
# Saving the Model #
####################

# We save the model with joblib
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../models/mnb.pkl')

joblib.dump(mnb, filename)
