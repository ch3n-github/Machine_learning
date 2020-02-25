#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv',delimiter=',')
x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]
n = x_data.shape
X_data = np.concatenate((np.ones(n),x_data),axis=1)


# In[22]:


def normal(xarr,yarr):
    x_data = np.mat(xarr)
    y_data = np.mat(yarr)
    xTx = x_data.T*x_data
    if np.linalg.det(xTx) == 0:
        print('此矩阵无逆矩阵')
        return
    w = xTx.I*x_data.T*y_data
    return w


# In[26]:


w = normal(X_data,y_data)
plt.scatter(x_data,y_data)
plt.plot(x_data,w[0,0]+w[1,0]*x_data,c='r')
plt.show


# In[18]:


np.mat(X_data.T)*np.mat(X_data)


# In[ ]:




