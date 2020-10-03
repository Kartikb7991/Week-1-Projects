#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


TT=pd.read_csv("titanic_train.csv")


# In[29]:


TT


# In[84]:


TT.describe()


# In[30]:


print("CHECKING FOR NULL VALUES:-")
TT.isnull()


# In[31]:


sns.heatmap(TT.isnull(),yticklabels=False)


# In[32]:


print("To show the count of people who survived and people who did not survive, i have used the countplot which is as follows:-")
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=TT)


# In[33]:


print("To show further classification based on gender count of people who survied and thos who did not , i have used to hue parameter which is as follows")
sns.countplot(x='Survived',hue='Sex',data=TT)


# In[34]:


sns.countplot(x='Survived',hue='Pclass',data=TT)


# In[35]:


sns.countplot(x='SibSp',data=TT)


# In[36]:


TT['Age'].mean()


# In[37]:


sns.boxplot(x='Pclass',y='Age',data=TT)


# In[38]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        
        elif Pclass==2:
            return 29
        
        else:
            return 24
        
    else:
        return Age
    


# In[39]:


impute_age


# In[40]:


TT['Age']=TT[['Age','Pclass']].apply(impute_age,axis=1)


# In[46]:


sns.heatmap(TT.isnull())


# In[47]:


TT.drop('Cabin',axis=1,inplace=True)


# In[48]:


sns.heatmap(TT.isnull())


# In[51]:


pd.get_dummies(TT['Embarked'],drop_first=True).head()


# In[52]:


sex=pd.get_dummies(TT["Sex"],drop_first=True)
Embarked=pd.get_dummies(TT['Embarked'],drop_first=True)


# In[55]:


TT.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[56]:


TT.head()


# In[58]:


TT=pd.concat([TT,sex,Embarked],axis=1)


# In[59]:


TT


# In[60]:


TT.drop('Survived',axis=1).head()


# In[61]:


TT['Survived'].head()


# In[62]:


from sklearn.model_selection import train_test_split


# In[67]:


x_train, x_test, y_train, y_test=train_test_split(TT.drop('Survived',axis=1),
                                                   TT['Survived'],test_size=0.30,
                                                   random_state=101)


# In[68]:


from sklearn.linear_model import LogisticRegression


# In[70]:


LR=LogisticRegression()
LR.fit(x_train,y_train)


# In[71]:


predictions=LR.predict(x_test)


# In[75]:


from sklearn.metrics import confusion_matrix


# In[76]:


accuracy=confusion_matrix(y_test,predictions)


# In[77]:


accuracy


# In[78]:


from sklearn.metrics import accuracy_score


# In[79]:


accuracy=accuracy_score(y_test,predictions)


# In[80]:


accuracy


# In[81]:


predictions


# In[ ]:




