#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


from sklearn.datasets import load_iris
iris=load_iris()
dir(iris)


# In[18]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.drop(['sepal length (cm)','sepal width (cm)'],axis='columns',inplace=True)
df


# In[19]:


plt.scatter(df['petal length (cm)'],df['petal width (cm)'])


# In[37]:


km=KMeans(n_clusters=2)
y_predicted=km.fit_predict(df[['petal length (cm)','petal width (cm)']])
y_predicted


# In[38]:


df['cluster']=y_predicted
df


# In[ ]:





# In[46]:


df0=df[df.cluster==0]
df1=df[df.cluster==1]


plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='pink')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='yellow')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker='*',color='red')

plt.xlabel('petal Length')
plt.ylabel('Petal width')
plt.legend()


# In[35]:


k_rang=range(1,10)
sse=[]
for k in k_rang:
    km= KMeans(n_clusters=k)
    km.fit(df[['petal length (cm)','petal width (cm)']])
    sse.append( km.inertia_)
sse


# In[36]:


plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_rang,sse)

