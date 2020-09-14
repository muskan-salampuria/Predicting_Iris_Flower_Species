"""
Iris petal Dataset Problem solved
usign Logistic Regression
Multiclassification
Data set is taken from sklearn.datasets
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

petals=load_iris()

print(dir(petals))

print(petals.data[0])
print(petals.feature_names[0])
print(petals.target[0])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(petals.data,petals.target,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train,y_train)

print(model.score(X_test,y_test))

y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_predicted)

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')