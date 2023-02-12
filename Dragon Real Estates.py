#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate - Price Predictor

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing = pd.read_excel(r"C:\Users\gupta\Downloads\Housing.xlsx")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts() #0->471,1->35


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


housing.hist(bins=50,figsize=(20,15))
plt.show()


# ## Train-Test Spliting

# In[10]:


# import numpy as np
# def split_train_test(data,test_ratio):
#     np.ramdom(42)
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data)*test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]


# In[11]:


# train_set,test_set = split_train_test(housing,0.2)


# In[12]:


# print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[15]:


strat_test_set


# In[16]:


strat_test_set.describe()


# In[17]:


strat_test_set['CHAS'].value_counts()


# In[18]:


strat_train_set['CHAS'].value_counts()


# In[19]:


95/7


# In[20]:


376/28


# In[21]:


housing = strat_train_set.copy() # Crete the copy of train set and store in the variable housing and then go for correlation
# Here our data is small that is why we check all the data for correlation


# ## Looking for Correlation

# In[22]:


corr_matrix = housing.corr() # Creating the correlation matrix


# In[23]:


corr_matrix['MEDV'].sort_values(ascending=False) # 1 means strong correlation


# ## Ploting the graphs !!!!

# In[24]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[25]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# ## Trying Out Attribute Contribution

# In[26]:


housing["TAXRM"] = housing['TAX']/housing['RM']


# In[27]:


housing.head()


# In[28]:


corr_matrix = housing.corr() # Creating the correlation matrix
corr_matrix['MEDV'].sort_values(ascending=False) # 1 means strong correlation


# In[29]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)


# In[30]:


housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing Attributes

# In[31]:


# To take care of miising attriburtes,you have three options
# 1-> Get rid of missing attribute
# 2-> Get rid of whole attribute
# 3-> Set the value to (0,mean,or median)


# In[32]:


a = housing.dropna(subset=["RM"]) # Option1
a.shape


# In[33]:


housing.drop("RM",axis=1) #Option2
# Note that there is no RM column and also note that the original housing data frame will remain unchanged


# In[34]:


median = housing["RM"].median()
median


# In[35]:


housing["RM"].fillna(median) # Compute median for option 3
# Note that there is no RM column and also note that the original housing data frame will remain unchanged


# In[36]:


housing.shape


# In[37]:


housing.describe() # Before we started filling missing values


# In[38]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[39]:


imputer.statistics_


# In[40]:


X = imputer.transform(housing)


# In[41]:


housing_tr = pd.DataFrame(X,columns=housing.columns) # Now we do not have any missing values


# In[42]:


housing_tr.describe()


# ## Scikit-learn Design

# Primarly,three types of objects
# 1-> Estimators -> It estimates some parameters based on a dataset.Eg imputer.
# It has fit method and transform method.
# Fit Method-> Fits the dataset and calculate internal parameters
# 
# 2-> Tranformers-> Transform method takes input and returns output based on the learning from fit().It also has a convenience function called fit_transform() which fits and then tranforms.
# 
# 3-> Predictors->Linear Regreesion model is an example of a predictor.fit() and predict() are two common functions.It also gives score() function which will evaluate the predictions.

# ## Feature Scaling

# Primarily,two types of feature scaling methods:-
# 1-> Min-Max scaling (Normalization)
# =(value-min)/(max-min)
# Sklearn provides a class called MinMaxScaler for this
# 
# 2-> Standarization
# =(value-mean)/std
# Sklearn provides a class called Standard Scaler for this

#    ## Creating the Pipeline

# In[43]:


from sklearn.pipeline import Pipeline
# We want to label the attributes (Taking all the label on the same scale)
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),  #.....add as many as you want
     ('std_scaler',StandardScaler()),
])


# In[44]:


housing_num_tr = my_pipeline.fit_transform(housing) # We will fit our pipeling in housing


# In[45]:


housing_num_tr.shape


# ## Selecting a desired model for Dragon Real Estates

# In[46]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[47]:


some_data = housing.iloc[:5]


# In[48]:


some_labels = housing_labels.iloc[:5]


# In[49]:


prepared_data = my_pipeline.transform(some_data)


# In[50]:


model.predict(prepared_data) # Will give an prediction array


# In[51]:


list(some_labels)


# ## Evaluating the model

# In[52]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# lin_mse # Now comparing the output of these line and label and we see that error is high so
# #We change the model from Linear Regression to Decision Tree

# In[53]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[54]:


#1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)


# In[55]:


rmse_scores


# In[56]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation",scores.std())


# In[57]:


print_scores(rmse_scores)


# Quiz:- Convert this notebook into a python file and run the pipeline using Visual Studio Code

# ## Saving the model

# ## Finally we chooses Random Forest Regressor Model

# In[58]:


from joblib import dump,load
dump(model,'Dragon.joblib')


# ## Testing the model on test data

# In[59]:


X_test = strat_test_set.drop("MEDV",axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions,list(Y_test))


# In[60]:


final_rmse


# In[61]:


prepared_data[0]


# ## Using the model for prediction

# In[62]:


from joblib import dump,load
import numpy as np
model = load('Dragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.25288536, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




