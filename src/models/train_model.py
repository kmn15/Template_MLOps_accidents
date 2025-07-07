import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np
import re

print(joblib.__version__)

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

def clean_float_string(s):
    # Remove all non-digit and non-period characters
    s = re.sub(r'[^\d.]', '', str(s))
    # If more than one period, keep only the last as decimal
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    # If empty or just periods, return np.nan
    if s.strip('.') == '':
        return np.nan
    return float(s)

for col in X_train.columns:
    X_train[col] = X_train[col].apply(clean_float_string)
for col in X_test.columns:
    X_test[col] = X_test[col].apply(clean_float_string)

rf_classifier = ensemble.RandomForestClassifier(n_jobs = -1, n_estimators=200, criterion='entropy')

#--Train the model
rf_classifier.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/trained_model.joblib'
joblib.dump(rf_classifier, model_filename)
print("Model trained and saved successfully.")
