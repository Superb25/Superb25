#!/usr/bin/env python
# coding: utf-8

# In[1]:


##import libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


banking_churn = pd.read_csv('https://raw.githubusercontent.com/regan-mu/ADS-April-2022/main/Assignments/Assignment%202/banking_churn.csv')
banking_churn.info()


# In[3]:


banking_churn.isna().sum()


# In[4]:


banking_churn.head()


# In[5]:


banking_churn.tail()


# In[6]:


print(banking_churn['EstimatedSalary'])


# In[7]:


#creating x variable
x = banking_churn.drop('EstimatedSalary', axis=1)
x.head()


# In[8]:


x.info()


# In[9]:


#creating y variable
y = banking_churn['EstimatedSalary']
y.head()


# In[46]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


# In[47]:


# Modelling
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100)

#model.fit(x_train, y_train)
#model.score(x_test, y_test)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()


# In[48]:


#spliting the data into training and test data
from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50,)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[49]:


# Turn the categories (Surname Geography and Gender) into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[60]:



# Define different features and transformer pipelines
categorical_features = ["Surname", "Geography", "Gender"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])


# In[62]:


#"RowNumber", "CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"]

categorical_features = ["Surname", "Geography", "Gender"]
hot = OneHotEncoder()
transformer = ColumnTransformer(transformers = [("one_hot", 
                                 hot, 
                                 categorical_features)],
                                 remainder = "passthrough")


transformed_x_train = transformer.fit_transform(x_train)
transformed_x_test = transformer.transform(x_test)


# In[51]:


print(transformed_x_train)
print(transformed_x_test)


# In[63]:


# Create a preprocessing and modelling pipeline
Model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("model", RandomForestRegressor())])
print(Model)


# In[65]:


#RandomforestClassifier
pipe = Pipeline({
    ('scaler', StandardScaler()),
    ('Random Forest', RandomForestClassifier())
    
})


# In[66]:


print(pipe)


# In[69]:


#Logistic Regression
pipe_LR = Pipeline({
    ('scaler', StandardScaler()),
    ('Logistic Regression', LogisticRegression())
    
})


# In[70]:


print(pipe_LR)


# In[71]:


sc = StandardScaler(with_mean=False)
x_train_sc = sc.fit(transformed_x_train)
X_train = x_train_sc.transform(transformed_x_train)
X_test = x_train_sc.transform(transformed_x_test)


# In[72]:


pipe.fit(transformed_x_train, y_train)


# In[73]:


from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

#view transformed values
print(y_transformed)


# In[74]:


transformed_x_train.shape


# In[ ]:


y_transformed.shape


# In[ ]:


clf = RandomForestClassifier()
clf.fit(transformed_x_train, y_transformed)


# In[ ]:


train = pd.DataFrame(transformed_x_train)
#train.columns = X_columns
train.head()


# In[ ]:


# Fit and score the model
model.fit(transformed_x_train, y_train)
model.score(transformed_x_test, y_test)


# In[ ]:


# Make predictions
y_preds = model.predict(transformed_x_test)

