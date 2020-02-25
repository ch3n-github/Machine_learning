#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts


# In[2]:


data = np.genfromtxt('linear.csv',delimiter=',')
x_data = data[1:,0,np.newaxis]
y_data = data[1:,1,np.newaxis]

plt.scatter(x_data,y_data)
plt.show


# In[3]:


x_train,x_test,y_train,y_test = tts(x_data,y_data,test_size = 0.2)


# In[4]:


model = linear_model.LinearRegression()
model.fit(x_train,y_train)


# In[5]:


plt.scatter(x_data,y_data)
plt.plot(x_data,model.predict(x_data),c='r')
plt.show


# In[6]:


model.score(x_test,y_test)


# In[7]:


model.coef_


# In[ ]:




