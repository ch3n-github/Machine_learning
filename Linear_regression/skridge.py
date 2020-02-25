#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[19]:


data = np.genfromtxt('longley.csv',delimiter=',')


# In[20]:


x_data = data[1:,2:]
y_data = data[1:,1]


# In[21]:


alpha_to_test = np.linspace(0.001,1)
model = linear_model.RidgeCV(alphas=alpha_to_test,store_cv_values=True)
model.fit(x_data,y_data)


# In[22]:


model.alpha_


# In[30]:


plt.plot(alpha_to_test,model.cv_values_.mean(axis=0))
plt.plot(model.alpha_,min(model.cv_values_.mean(axis=0)),'ro')
plt.show


# In[31]:


model.coef_


# In[ ]:




