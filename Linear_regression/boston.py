#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV


# In[5]:


house = load_boston()

x_data = house.data
y_data = house.target

df = pd.DataFrame(x_data,columns=house.feature_names)
df['Target']=pd.DataFrame(y_data,columns=['Target'])
df


# In[10]:


plt.figure(figsize=(15,15))
heatmap = sns.heatmap(df.corr(),annot=True,square=True)


# In[44]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_data = ss.fit_transform(x_data)

x_train,x_test,y_train,y_test = tts(x_data,y_data,test_size=0.5)
model = LassoCV()
model.fit(x_train,y_train)


# In[45]:


print(model.alpha_)
print(model.coef_)


# In[46]:


model.score(x_test,y_test)


# In[ ]:




