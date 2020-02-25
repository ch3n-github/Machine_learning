#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = np.genfromtxt('longley.csv',delimiter=',')
x_data = data[1:,2:]
y_data = data[1:,1,np.newaxis]
print(x_data,y_data)


# In[7]:


X_data = np.concatenate((np.ones((len(x_data),1)),x_data),axis=1)
def weight(x_data,y_data,lamda):
    x_mat=np.mat(x_data)
    y_mat=np.mat(y_data)
    chan=np.mat(np.eye(len(x_mat.T*x_mat))*lamda)
    return (x_mat.T*x_mat+chan).I*x_mat.T*y_mat
ws = weight(X_data,y_data,0.4)


# In[10]:


print(ws)


# In[9]:


print(np.mat(X_data)*ws)


# In[ ]:




