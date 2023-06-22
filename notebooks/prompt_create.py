#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[9]:


df = pd.read_excel('data/novel_sentence.xlsx', header=None)


# In[13]:


df


# In[14]:


for i in range(10):
    print("novel need: " + df[0].iloc[i])

