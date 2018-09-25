
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[37]:


diabetes = datasets.load_diabetes()


# In[18]:



# Use only one feature# Use o 
diabetes_X = diabetes.data[:, np.newaxis, 2]
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]


# In[19]:


# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]


# In[20]:


# Create linear regression object
regr = linear_model.LinearRegression()


# In[21]:


# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)


# In[22]:


# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)


# In[23]:


# The coefficients
print('Coefficients: \n', regr.coef_)


# In[24]:


print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))


# In[25]:


# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))


# In[26]:


# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())


# In[27]:


plt.show()

