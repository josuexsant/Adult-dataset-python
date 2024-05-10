import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier

import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

import scipy.stats as stats


names_col = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
       'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
       'hours_per_week', 'native_country', 'income']

df = pd.read_csv("data/adult.csv", names=names_col)

from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(data):
  le = LabelEncoder()
  for column in data.columns:
    if data[column].dtype == type(object):
      data[column] = le.fit_transform(data[column])
  return data


def preprocess_data(df):
  df.drop_duplicates(inplace=True)
  df.replace('?', np.nan, inplace=True)
  df.dropna(inplace=True)
  z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
  threshold = 3
  df = df[(z_scores < threshold).all(axis=1)]
  ros = RandomOverSampler()
  X_resampled, y_resampled = ros.fit_resample(df.drop('income', axis=1), df['income'])
  df = pd.concat([X_resampled, y_resampled], axis=1)


  def apply_j48(df):
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return clf, f1, accuracy, report

  def apply_svm(df):
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = SVC()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return clf, f1, accuracy, report

  def apply_knn(df):
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return clf, f1, accuracy, report

  def apply_naive_bayes(df):
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return clf, f1, accuracy, report

  j48_model, j48_f1, j48_accuracy, j48_report = apply_j48(df)
  svm_model, svm_f1, svm_accuracy, svm_report = apply_svm(df)
  knn_model, knn_f1, knn_accuracy, knn_report = apply_knn(df)
  nb_model, nb_f1, nb_accuracy, nb_report = apply_naive_bayes(df)
  
  print("J48 Report:")  
  print(j48_report)

  print("SVM Report:")
  print(svm_report)

  print("KNN Report:")
  print(knn_report)

  print("Naive Bayes Report:")
  print(nb_report)
  
  
df = encode_categorical_columns(df)
preprocess_data(df)
 
  
  