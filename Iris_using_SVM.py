#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()


# In[2]:


dir(iris)


# In[3]:


iris.feature_names


# In[4]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[5]:


df['target']=iris.target


# In[6]:


df[df.target==1].head()


# In[7]:


df[df.target==2].head()


# In[8]:


df[df.target==0].head()


# In[9]:


df['flower_name']=df.target.apply(lambda x: iris.target_names[x])


# In[10]:


df.head()


# In[11]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
df0.head()


# In[13]:


plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='red',marker='.')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')


# In[14]:


plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green',marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='red',marker='.')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')


# In[15]:


X=df.drop(['target','flower_name'],axis='columns')
y=df.target


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(X,y)


# In[18]:


from sklearn.svm import SVC
model=SVC(kernel='linear',gamma=100,C=1)
model.fit(X_train,y_train)


# In[20]:


print(model.score(X_test,y_test))


# In[ ]:





# In[ ]:




