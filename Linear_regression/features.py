#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


data = np.genfromtxt('job.csv',delimiter=',')


# In[3]:


x_data = data[1:,1]
y_data = data[1:,2]
plt.scatter(x_data,y_data)
plt.show


# In[4]:


model = LinearRegression()
nx_data = x_data[:,np.newaxis]
ny_data = y_data[:,np.newaxis]
model.fit(nx_data,ny_data)


# In[5]:


plt.scatter(x_data,y_data)
plt.plot(nx_data,model.predict(nx_data),c='r')
plt.show


# In[28]:


plot_reg = PolynomialFeatures(degree=6)
x_poly = plot_reg.fit_transform(nx_data)
model2 = LinearRegression()
model2.fit(x_poly,ny_data)


# In[29]:


plt.scatter(x_data,y_data)
plt.plot(nx_data,model2.predict(x_poly),c='r')
plt.show


# In[19]:





# In[ ]:




