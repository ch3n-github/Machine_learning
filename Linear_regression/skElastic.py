#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


data = np.genfromtxt('longley.csv',delimiter=',')
print(data)


# In[3]:


x_data = data[1:,2:]
y_data = data[1:,1,np.newaxis]
print(x_data,y_data)


# In[6]:


model= linear_model.ElasticNetCV()
model.fit(x_data,y_data)

print(model.alpha_)
print(model.coef_)


# In[7]:


model.predict(x_data)


# In[ ]:




