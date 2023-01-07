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


# In[12]:


# Diagonalzing the random matrix.
w, v = LA.eigh(X)


# In[13]:


# Function calculate level statistics.
def Level_Statistics(n,Es):
    return min(abs(Es[n]-Es[n-1]),abs(Es[n+1]-Es[n]))/max(abs(Es[n]-Es[n-1]),abs(Es[n+1]-Es[n]))


# In[14]:


def KLd(Eigenvector_matrix):
    KL = []
    for n in range(2**L-1): # Eigenvector index goes from 0 to dim(H)-1.
        KLd_sum = 0.0
        for i in range(2**L): # The sum goes from 0 to dim(H) i.e length of an eigenvector.
            p = LA.norm(Eigenvector_matrix[0:2**L,n][i])**2 + 1.e-9
            q = LA.norm(Eigenvector_matrix[0:2**L,n+1][i])**2 + 1.e-9
            KLd_sum += p*(np.log(p/q))
        KL.append(KLd_sum)
    return KL


# In[15]:


f_KLd = open('KLd_data'+'.txt', 'w')
KLd_calculated = KLd(v)
for k in range(2**L-1):
    f_KLd = open('KLd_data'+'.txt', 'a')
    f_KLd.write(str(k) + '\t'+ str(KLd_calculated[k]) +'\n')
f_KLd.close()


# In[16]:

r"""
f = open('level_statistics_data'+'.txt', 'w')
for i in range(1,2**L-1):
    f = open('level_statistics_data'+'.txt', 'a')
    f.write(str(i) + '\t'+ str(Level_Statistics(i,w)) +'\n')
f.close()"""

