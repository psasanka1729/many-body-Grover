#!/usr/bin/env python
# coding: utf-8

# # Calculation of level statistics and KL divergence of GUE

# In[9]:


import numpy as np
from numpy import linalg as LA
import tenpy


# In[10]:


# Number of qubits.
L = 10


# In[11]:


# Generating a random GUE.
X = tenpy.linalg.random_matrix.GUE((2**L,2**L))

np.save('gue.npy',X)
