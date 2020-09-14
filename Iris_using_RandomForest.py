#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()


# In[4]:


dir(iris)


# In[5]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)


# In[6]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,iris.target,test_size=0.2)


# In[22]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)


# In[23]:


model.score(X_test,y_test)


# In[24]:


y_predicted=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predicted)


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




