#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[3]:


data = np.genfromtxt('longley.csv',delimiter=',')
x_data = data[1:,2:]
y_data = data[1:,1,np.newaxis]
print(x_data,y_data)


# In[4]:


model = linear_model.LassoCV()
model.fit(x_data,y_data)


# In[5]:


#指标参数
print(model.coef_)
#LASSO系数
print(model.alpha_)


# In[7]:


print(y_data,model.predict(x_data))


# In[8]:


model.score(x_data,y_data)


# In[ ]:




