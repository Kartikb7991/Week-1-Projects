#!/usr/bin/env python
# coding: utf-8

# In[198]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[199]:


WC=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\wine-quality.csv')


# In[201]:


WC


# In[202]:


WC.shape


# In[203]:


WC.dtypes


# In[204]:


WC.columns


# In[205]:


WC.describe()


# In[206]:


WC.quality.unique()


# In[207]:


WC.quality.value_counts()


# In[208]:


sns.heatmap(WC.isnull())


# In[209]:


WC.corr()


# In[210]:


sns.heatmap(WC.corr())


# In[211]:






# In[212]:


collist=WC.columns.values
ncol=12
nrows=10


# In[213]:


plt.figure(figsize=(ncol,7*ncol))
for i in range(1, len(collist)):
    plt.subplot(nrows,ncol,i+1)
    sns.boxplot(WC[collist[i]],color='green',orient='v')
    plt.tight_layout


# In[214]:


plt.figure(figsize=(16,16))
for i in range(0,len(collist)):
    plt.subplot(nrows,ncol,i+1)
    sns.distplot(WC[collist[i]])


# In[215]:


WC.drop('volatile acidity',axis=1,inplace=True)


# In[216]:


WC.head()


# In[217]:


from scipy.stats import zscore
z=np.abs(zscore(WC))
z


# In[218]:


threshold=3
print(np.where(z>3))


# In[219]:


WCN=WC[(z<3).all(axis=1)]


# In[220]:


WCN


# In[221]:


WCN.shape


# In[222]:


bins=[0,5.5,7.7,10]
labels=[0,1,2]
WCN['quality']=pd.cut(WCN['quality'],bins=bins,labels=labels)


# In[223]:


WCN.head()


# In[224]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[225]:


WCN['quality'].unique()


# In[226]:


x=WCN[WCN.columns[:-1]]
y=WCN['quality']
sc=StandardScaler()
x=sc.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


# In[ ]:





# In[234]:


n5=KNeighborsClassifier(n_neighbors=5)
n5.fit(x_train,y_train)
n5.score(x_train,y_train)
pred_n5=n5.predict(x_test)
print(classification_report(y_test,pred_n5))
print(accuracy_score(y_test,pred_n5))
print(confusion_matrix(y_test,pred_n5))


# In[ ]:




