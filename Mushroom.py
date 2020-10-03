#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[67]:


data=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\mushroom.csv')


# In[68]:


data


# In[69]:


data.columns


# In[70]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[71]:


mappings=list()
le=LabelEncoder()


# In[116]:


data.shape


# In[118]:


data.dtypes


# In[119]:


sns.heatmap(data.corr())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[73]:


data.CapShape=le.fit_transform(data.CapShape)
data.CapSurface=le.fit_transform(data.CapSurface)
data.CapColor=le.fit_transform(data.CapColor)
data.Bruises=le.fit_transform(data.Bruises)
data.Odor=le.fit_transform(data.Odor)
data.GillAttachment=le.fit_transform(data.GillAttachment)
data.GillSpacing=le.fit_transform(data.GillSpacing)
data.GillSize=le.fit_transform(data.GillSize)
data.GillColor=le.fit_transform(data.GillColor)
data.StalkShape=le.fit_transform(data.StalkShape)
data.StalkRoot=le.fit_transform(data.StalkRoot)
data.StalkSurfaceAboveRing=le.fit_transform(data.StalkSurfaceAboveRing)
data.StalkSurfaceBelowRing=le.fit_transform(data.StalkSurfaceBelowRing)
data.StalkColorAboveRing=le.fit_transform(data.StalkColorAboveRing)
data.StalkColorBelowRing=le.fit_transform(data.StalkColorBelowRing)
data.VeilType=le.fit_transform(data.VeilType)
data.VeilColor=le.fit_transform(data.VeilColor)
data.RingNumber=le.fit_transform(data.RingNumber)
data.RingType=le.fit_transform(data.RingType)
data.SporePrintColor=le.fit_transform(data.SporePrintColor)
data.Population=le.fit_transform(data.Population)
data.Habitat=le.fit_transform(data.Habitat)
data.Class=le.fit_transform(data.Class)


# In[74]:


data.head()


# In[75]:


df.isnull().sum()


# In[76]:


data.describe()


# In[78]:


sns.heatmap(data.isnull(),yticklabels=False)


# In[80]:


print("To show the count of mushrooms which were edible and ones which were poisonous, i have used the countplot which is as follows:-")
sns.set_style('whitegrid')
sns.countplot(x='Class',data=data)


# In[81]:


sns.countplot(x='Class',hue='CapShape',data=data)


# In[83]:


sns.countplot(x='Class',hue='Population',data=data)


# In[84]:


sns.countplot(x='RingType',data=data)


# In[87]:


sns.boxplot(x='CapColor',y='CapShape',data=data)


# In[91]:


y=data['Class']
x=data.drop('Class',axis=1)


# In[93]:


x.shape


# In[94]:


y.shape


# In[95]:


x


# In[96]:


y


# In[105]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)


# In[106]:


LR=LogisticRegression()
SV=SVC(C=1.0,kernel='rbf')


# In[109]:


LR.fit(x_train,y_train)


# In[111]:


LR.score(x_test,y_test)


# In[112]:


print('Logistic Regression gives us 95 percent accuracy')


# In[113]:


SV.fit(x_train,y_train)


# In[114]:


SV.score(x_test,y_test)


# In[127]:


print(' svm gives us 99 percent accuracy')


# In[ ]:




