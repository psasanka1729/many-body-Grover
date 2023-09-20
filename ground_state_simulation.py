#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tenpy
import numpy as np
from numpy import linalg as LA


# In[8]:


Delta = 1


# In[9]:


def bra_0(m):
    bra_0 = np.matrix(np.zeros(m))
    bra_0[0,0] = 1
    return bra_0


# In[10]:


def H_0(m):
    H_matrix = np.zeros((m,m))
    H_matrix[0,0] = 1
    return H_matrix


# In[11]:


def V(m):
    standard_GUE = tenpy.linalg.random_matrix.GUE((m,m))/np.sqrt(m)
    return standard_GUE-(1/m)*np.trace(standard_GUE)*np.identity(m)


# In[12]:


def H(m,lam):
    return -Delta*H_0(m)+lam*V(m)


# In[13]:


def ground_state(hamiltonian):
    eigenvalues, eigenvectors = LA.eigh(hamiltonian)
    min_eigenvalue_index = np.argmin(eigenvalues)
    return np.matrix(eigenvectors[:, min_eigenvalue_index]).T


# In[14]:


def overlap(m,lam):
    psi = ground_state(H(m,lam))
    return np.abs((bra_0(m)@psi)[0,0])**2/np.abs((psi.conj().T@psi)[0,0])**2


# In[15]


# In[ ]:

lamb_input = float(sys.argv[1])

file_to_write = open("overlap.txt","w")
# overlap for N = 100
file_to_write.write( str(overlap(4000,lamb_input)) + "\t" +
                     str(overlap(6000,lamb_input)) + "\t" +
                     str(overlap(8000,lamb_input)) + "\t" +
                     str(overlap(10000,lamb_input)) + "\t" +
                     str(overlap(12000,lamb_input))
                     )
file_to_write.close()
#pf.write(str(i+1)+'\t'+ str(r_avg[0,i]) +'\t' + str(r_SE[i])+'\n')       


# In[ ]:




