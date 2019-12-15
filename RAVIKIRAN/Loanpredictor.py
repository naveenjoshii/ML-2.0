#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install missingno


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv('loan.csv')


# In[4]:


data.head(3)


# In[5]:


data.info()


# In[6]:


ms.matrix(data)


# In[7]:


Gender_df = pd.get_dummies(data['Gender'],drop_first=3)
Gender_df.head()


# In[8]:


Married_df = pd.get_dummies(data['Married'],drop_first=3)
Married_df.head()


# In[9]:


Education_df = pd.get_dummies(data['Education'],drop_first=3)
Education_df.head()


# In[10]:


Self_Employed_df = pd.get_dummies(data['Self_Employed'],drop_first=3)
Self_Employed_df.head()


# In[11]:


Property_Area_df = pd.get_dummies(data['Property_Area'],drop_first=3)
Property_Area_df.head()


# In[12]:


Loan_Status_df = pd.get_dummies(data['Loan_Status'],drop_first=3)
Loan_Status_df.head()


# In[13]:


old_data = data.copy()
data.drop(['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status'],axis=1,inplace=True)
data.head()


# In[14]:


data = pd.concat([data,Gender_df,Married_df,Education_df,Self_Employed_df,Property_Area_df,Loan_Status_df],axis=1)


# In[15]:


data.head()


# In[16]:


data.columns = ['Loan_ID', 'Dependents', 'ApplicantIncome', 'CoapplicantIncome', 
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Gender','Married','Education','Self_Employed','Semiurban','Urban','Loan_Status']


# In[17]:


ms.matrix(data)


# In[20]:


def impute_Dependents(cols):
    Dependents = cols[0]
    
    if pd.isnull(Dependents):
        return 0

    else:
        return Dependents


# In[22]:


data['Dependents'] = data[['Dependents']].apply(impute_Dependents,axis=1)


# In[23]:


ms.matrix(data)


# In[24]:


def impute_Loan_Amount_Term(cols):
    Loan_Amount_Term = cols[0]
    
    if pd.isnull(Loan_Amount_Term):
        return 360.0

    else:
        return Loan_Amount_Term


# In[25]:


data['Loan_Amount_Term'] = data[['Loan_Amount_Term']].apply(impute_Loan_Amount_Term,axis=1)


# In[26]:


ms.matrix(data)


# In[27]:


sns.heatmap(data.corr(),cmap='coolwarm',xticklabels=True,annot=True)
plt.title('data.corr()')


# In[28]:


data.corr()


# In[30]:


data.groupby('Self_Employed')['LoanAmount'].median()


# In[31]:


def impute_LoanAmount(cols):
    LoanAmount= cols[0]
    Self_Employed= cols[1]
    
    if pd.isnull(LoanAmount):
        # Class-0
        if Self_Employed == 0:
            return 125.0
        else:
            return 150.0

    else:
        return LoanAmount


# In[34]:


data['LoanAmount'] = data[['LoanAmount','Self_Employed']].apply(impute_LoanAmount,axis=1)


# In[35]:


ms.matrix(data)


# In[36]:


data.groupby('Loan_Status')['Credit_History'].median()


# In[37]:


def impute_Credit_History(cols):
    Credit_History = cols[0]
    
    if pd.isnull(Credit_History):
        return 1.0

    else:
        return Credit_History


# In[38]:


data['Credit_History'] = data[['Credit_History']].apply(impute_Credit_History,axis=1)


# In[39]:


ms.matrix(data)


# In[51]:


data.drop(['Dependents'],axis=1,inplace=True)


# In[52]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('Loan_Status',axis=1), 
                                                    data['Loan_Status'], test_size=0.30, 
                                                    random_state=101)


# In[54]:


from sklearn.linear_model import LogisticRegression

# Build the Model.
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[55]:


predicted =  logmodel.predict(X_test)


# In[56]:


predicted


# In[57]:


from sklearn.metrics import precision_score

print(precision_score(y_test,predicted))


# In[58]:


from sklearn.metrics import recall_score

print(recall_score(y_test,predicted))


# In[59]:


from sklearn.metrics import f1_score

print(f1_score(y_test,predicted))


# In[60]:


from sklearn.metrics import classification_report

print(classification_report(y_test,predicted))

