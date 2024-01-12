#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


iris=pd.read_csv("Downloads/Iris.csv")


# In[3]:


iris.head()


# In[4]:


iris.shape


# In[5]:


iris.describe()


# In[6]:


iris.info()


# In[7]:


iris.isnull().sum()


# In[8]:


iris.mean()


# In[9]:


iris.min()


# In[10]:


iris.max()


# In[11]:


def half(s):
    return s*0.5


# In[12]:


iris[['SepalLengthCm','PetalLengthCm']].apply(half)


# In[13]:


iris.drop('SepalLengthCm',axis=1)


# In[14]:


iris['Species'].value_counts()


# In[15]:


sns.scatterplot(x="SepalLengthCm",y="PetalLengthCm",data=iris,hue="Species",style="Species")


# In[16]:


X=iris.values[:,0:-1]
Y=iris.values[:,-1]


# In[17]:


print(X.shape)
print(Y.shape)


# In[18]:


#Splitting the data into Train and Test
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=10)


# In[20]:


from sklearn.tree import DecisionTreeClassifier

model_DT=DecisionTreeClassifier(random_state=10,criterion='gini')
model_DT.fit(X_train, Y_train)
Y_pred=model_DT.predict(X_test)
print(Y_pred)


# In[21]:


print(list(zip(Y_test,Y_pred)))


# In[22]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test, Y_pred)
print(cfm)
print("Classification report: ")
print(classification_report(Y_test, Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# In[ ]:




