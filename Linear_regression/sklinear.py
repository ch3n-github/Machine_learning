#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt


# In[19]:


data = np.genfromtxt('data.csv',delimiter=',')
X = data[:,0]
Y = data[:,1]
plt.scatter(X,Y)
plt.show
data.shape


# In[21]:


x_data=data[:,0,np.newaxis]
y_data=data[:,1,np.newaxis]

model =LR()
model.fit(x_data,y_data)


# In[30]:


plt.scatter(x_data,y_data,color="b")
plt.plot(x_data,model.predict(x_data),"r")
plt.show


# In[ ]:




