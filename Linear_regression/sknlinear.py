#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression as LR


# In[5]:


data = np.genfromtxt("Delivery.csv",delimiter=",")
x_data = data[:,:-1]
y_data = data[:,-1]


# In[6]:


def compute_error(theta0,theta1,theta2,x1_data,x2_data,y_data):
    error = 0
    for i in range(len(x1_data)):
        error+=(theta0+theta1*x1_data[i]+theta2*x2_data[i]-y_data[i])**2
    return error/len(x1_data)


# In[7]:


model =LR()
model.fit(x_data,y_data)


# In[8]:


#系数
print(model.coef_)


# In[9]:


#截距
print(model.intercept_)


# In[12]:


fig = plt.figure()
ax = Axes3D(fig)
x1_data=data[:,0]
x2_data=data[:,1]
ax.scatter(x1_data,x2_data,y_data,c='r',s=100)
X1_data,X2_data=np.meshgrid(x1_data,x2_data)
Y_data=model.intercept_+model.coef_[0]*X1_data+model.coef_[1]*X2_data
ax.plot_surface(X1_data,X2_data,Y_data)
plt.show()


# In[ ]:




