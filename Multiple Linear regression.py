#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


sns.get_dataset_names()


# In[23]:


df=sns.load_dataset("tips")
df.head()


# In[24]:


df.shape


# In[25]:


df.info()      #Basic information about data set


# In[26]:


df.isna().sum()   #TO check the null values


# In[94]:


df.corr()


# In[99]:


sns.pairplot(df)


# In[27]:


sns.histplot(df["tip"],kde=True)


# In[28]:


sns.histplot(df["total_bill"],kde=True)


# In[53]:


sns.countplot(df["sex"])


# In[51]:


sns.countplot(df["smoker"])


# In[54]:


sns.countplot(df["day"])


# In[55]:


sns.countplot(df["time"])


# In[56]:


df["size"].unique()


# In[57]:


sns.countplot(df["size"])


# Let us consider the X: Independent variable and Y= "tip" dependent variable

# In independent features sex,smoker, day, and time are categorical variables

# In[134]:


X=df.drop(["tip"],axis = 1)
Y=df.iloc[:,1]
print(X.head())


# In[135]:


X=pd.get_dummies(X,drop_first=True)
X.head()


# In[136]:


print(X.shape,Y.shape)


# In[137]:


import sklearn


# In[138]:


from sklearn.model_selection import train_test_split


# In[139]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=2)


# In[140]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[141]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()


# In[142]:


model=reg.fit(x_train,y_train)


# In[143]:


model.intercept_


# In[144]:


model.coef_


# In[145]:


y_pred=model.predict(x_test)


# In[146]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred);score


# # Use full data set to get ANOVA table , Summary of regression model

# In[151]:


import statsmodels.api as sm


# In[152]:



X  #independent variable
Y  #dependent variable
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())


# **From above table we can say that "total_bill","Size" are significant at 6% level of significane**

# In[167]:


X_sign=X[["total_bill","size"]]


# In[168]:


X2 = sm.add_constant(X_sign)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:




