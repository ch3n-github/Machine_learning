#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts


# In[2]:


data = pd.read_csv('data.csv')
data = np.array(data)
X = data[:,0]
Y = data[:,1]
plt.scatter(X,Y)
plt.title('Salary data')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend
plt.show
data


# 

# In[142]:


print(np.shape(data))


# In[24]:


#学习效率
learning_rate = 0.0001

n = len(data)
b = 0
k = 0
epoch = 50

def func(theata0,theata1,x):
    return theata0+theata1*x

for j in range(epoch):
    det0 = 0
    det1 = 0
    for i in range(0,len(X)):
        det0 += (func(b,k,X[i])-Y[i])/n
        det1 += (X[i]*(func(b,k,X[i])-Y[i]))/n
    print(det0,det1)    
    b = b- learning_rate*det0
    k = k- learning_rate*det1

plt.scatter(X,Y)
plt.plot(X,b+k*X,'r')
b,k


# In[25]:


plt.scatter(X,Y)
plt.plot(X,b+k*X,'r')
plt.title('Salary data')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend
plt.show


# In[ ]:




