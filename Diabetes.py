#!/usr/bin/env python
# coding: utf-8

# ## Importing Dependencies

# In[91]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# ## Data Collection and Analysis
# 
# 

# In[92]:


diabetes_dataset = pd.read_csv('/Users/akilasivan/Desktop/Diabetes.csv')


# In[93]:


# number of rows and Columns in this dataset
diabetes_dataset.shape


# In[94]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[95]:


diabetes_dataset['Outcome'].value_counts()


# In[96]:


diabetes_dataset.groupby('Outcome').mean()


# In[97]:


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[98]:


print(X)


# In[99]:


print(Y)


# ### Data Standardization

# In[100]:


scaler = StandardScaler()


# In[101]:


scaler.fit(X)


# In[102]:


standardized_data = scaler.transform(X)


# In[103]:


print(standardized_data)


# In[104]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[105]:


print(X)
print(Y)


# ### Train Test Split

# In[106]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[107]:


print(X.shape, X_train.shape, X_test.shape)


# ### Training the Model

# In[108]:


classifier = svm.SVC(kernel='linear')


# In[109]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# ## Model Evaluation

# #### Accuracy Score

# In[110]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[111]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[112]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[113]:


print('Accuracy score of the test data : ', test_data_accuracy)


# #### Making a Predictive System

# In[114]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




