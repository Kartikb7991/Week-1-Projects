#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[44]:


df=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\sonar.csv')


# In[45]:


df


# In[46]:


df.shape


# In[47]:


df.columns


# In[48]:


x=df.drop('Class',1)
df['Class']=df['Class'].astype('category')
y=df['Class']


# In[49]:


df.isnull().sum()


# In[50]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=4)


# In[51]:


logreg=LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)


# In[52]:


y_pred=logreg.predict(x_test)
print(confusion_matrix(y_test,y_pred))


# In[53]:


print('Accuracy',metrics.accuracy_score(y_test,y_pred))


# In[54]:


from sklearn.svm import SVC


# In[55]:


svc=SVC()
svc.fit(x_train,y_train)
svc.score(x_train,y_train)
predsvc=svc.predict(x_test)
print(accuracy_score(y_test,predsvc))
print(confusion_matrix(y_test,predsvc))


# In[56]:


print('SVC WORKS BETTER FOR THIS MODEL WITH AN ACCURACY SCORE OF 83 PERCENT')


# In[ ]:




