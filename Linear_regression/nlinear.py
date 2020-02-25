#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


data = np.genfromtxt("Delivery.csv",delimiter=",")
x1_data = data[:,0]
x2_data = data[:,1]
y_data = data[:,2]


# In[3]:


def compute_error(theta0,theta1,theta2,x1_data,x2_data,y_data):
    error = 0
    for i in range(len(x1_data)):
        error+=(theta0+theta1*x1_data[i]+theta2*x2_data[i]-y_data[i])**2
    return error/len(x1_data)


# In[35]:


n = float(len(x1_data))

theta0=0
theta1=0
theta2=0

lr = 0.0001

epoch = 50

error0 =0

for i in range(epoch*1000):
    grad0 = 0
    grad1 = 0
    grad2 = 0
    for j in range(len(x1_data)):
        grad0 += (theta0+theta1*x1_data[j]+theta2*x2_data[j]-y_data[j])*2/n
        grad1 += x1_data[j]*(theta0+theta1*x1_data[j]+theta2*x2_data[j]-y_data[j])*2/n
        grad2 += x2_data[j]*(theta0+theta1*x1_data[j]+theta2*x2_data[j]-y_data[j])*2/n
    theta0 = theta0 -lr*grad0
    theta1 = theta1 -lr*grad1
    theta2 = theta2 -lr*grad2
    error1 = compute_error(theta0,theta1,theta2,x1_data,x2_data,y_data)
    if abs(error0-error1)<0.000001:
        break
    error0 = error1
def func(theta0,theta1,theta2,x1_data,x2_data):
    return theta0+theta1*x1_data+theta2*x2_data
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1_data,x2_data,y_data,c='r',s=100)
X1_data,X2_data=np.meshgrid(x1_data,x2_data)
Y_data=func(theta0,theta1,theta2,X1_data,X2_data)
ax.plot_surface(X1_data,X2_data,Y_data)
plt.show()


# In[33]:


print(theta0,theta1,theta2)
compute_error(theta0,theta1,theta2,x1_data,x2_data,y_data)


# In[ ]:




