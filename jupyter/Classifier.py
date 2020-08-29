# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from custom import preprocessing as pr
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# # Data preprocessing

mnist = pr.import_mnist()
X, y = mnist["data"], mnist["target"]
y = y.astype('int32')

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

# ## Saving the scaler

# +
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    
filename = "scaler"
with open(filename, 'wb') as output:  
    pickle.dump(scaler, output, pickle.HIGHEST_PROTOCOL)
# -

# # Stochastic gradient

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train_scaled, y_train)

sgd_score=cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

sgd_score

sgd_score.mean()

# ## Saving the model

import joblib
filename = 'sgd_clf.sav'
joblib.dump(sgd_clf, filename)

# # K nearest neighbours

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=1,weights="uniform", metric="cosine")
knn_clf.fit(X_train_scaled,y_train)

knn_score=cross_val_score(knn_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

knn_score

knn_score.mean()

# ## Saving the model

import joblib
filename = 'knn_1_uniform_cosine.sav'
joblib.dump(knn_clf, filename)

# # Evaluating on test set

model1 = joblib.load("knn_1_uniform_cosine.sav")

X_test_scaled = scaler.transform(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import matplotlib.pyplot as plt

predictions = model1.predict(X_test_scaled)
accuracy_score(y_test, predictions)

print (classification_report(y_test,predictions))

conf_mx = confusion_matrix(y_test, predictions)
conf_mx

# ## Confusion matrix plot

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# ## Error plot

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

# **Slightly worse performence than random forest, yet KNN with cosine metric proved to be clearly faster**

# # Random forest classifier

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier()
forest.fit(X_train_scaled,y_train)

forest_score=cross_val_score(knn_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

forest_score

forest_score.mean()

# ## Saving the model

import joblib
filename = 'forest.sav'
joblib.dump(forest, filename)

# # Evaluating on test set

model1 = joblib.load("forest.sav")

X_test_scaled = scaler.transform(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import matplotlib.pyplot as plt

predictions = model1.predict(X_test_scaled)
accuracy_score(y_test, predictions)

print (classification_report(y_test,predictions))

conf_mx = confusion_matrix(y_test, predictions)
conf_mx

# ## Confusion matrix plot

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# ## Error plot

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

# # Conclusions

# **Random forest proved to be the best with 97% accuracy score. Yet KNN was with cosine metric was visibly faster, with still high accuracy of 94%**
