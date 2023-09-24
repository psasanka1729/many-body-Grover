#!/usr/bin/env python
# coding: utf-8

# In[7]:

import sys
import tenpy
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from numpy import linalg as LA
#import matplotlib.pyplot as plt


# In[8]:


Delta = 1


# In[9]:


def bra_0(m):
    #bra_0 = np.matrix(np.zeros(m))
    bra_0 = scipy.sparse.lil_matrix((1, m),dtype = np.float64)
    bra_0[0,0] = 1
    return bra_0


# In[10]:


def H_0(m):
    H_matrix = np.zeros((m,m))
    H_matrix[0,0] = 1
    H_matrix = scipy.sparse.lil_matrix(H_matrix)
    return H_matrix


# In[11]:


def V(m):
    standard_GUE = tenpy.linalg.random_matrix.GUE((m,m))/np.sqrt(m)
    return scipy.sparse.lil_matrix(standard_GUE-(1/m)*np.trace(standard_GUE)*np.identity(m))


# In[12]:


def H(m,lam):
    return scipy.sparse.lil_matrix(-Delta*H_0(m)+lam*V(m))


# In[13]:


def ground_state(hamiltonian):
    eigenvalues, eigenvectors = eigsh(hamiltonian,k=2)
    #min_eigenvalue_index = np.argmin(eigenvalues)
    return eigenvalues,scipy.sparse.lil_matrix(eigenvectors)#np.matrix(eigenvectors[:, min_eigenvalue_index]).T


# In[14]:
lamb_input = float(sys.argv[1])


m = 17000
standard_GUE = tenpy.linalg.random_matrix.GUE((m,m))/np.sqrt(m)
V = scipy.sparse.lil_matrix(standard_matrix-(1/m)*np.trace(standard_GUE)*np.identity(m))
H = scipy.sparse.lil_matrix(-Delta*H_0(m)+lamb_input*V)
eigenvalues,eigenstates = eigsh(H,k = 2)
np.save("eigenstate",eigenstates[:,0])
np.save("energy_gap.npy",eigenvalues[1]-eigenvalues[0])


#def overlap(m,lam):
#	gr_state = ground_state(H(m,lam))
#	psi = gs_state[1][:,1]
#	energy_gap = gs_state[0][1]-gs_state[0][0]
#	return energy_gap,np.abs(bra_0(m).dot(psi).A[0,0])**2/np.abs(psi.conj().T.dot(psi)[0,0])**2
    #return np.abs((bra_0(m)@psi)[0,0])**2/np.abs((psi.conj().T@psi)[0,0])**2


# In[15]


# In[ ]:
r"""
lamb_input = float(sys.argv[1])

file_to_write = open("overlap.txt","w")
# overlap for N = 100
file_to_write.write( str(overlap(16000,lamb_input)) + "\t" +
                     str(overlap(17000,lamb_input)) + "\t" +
		     str(ei)
                     )
file_to_write.close()
#pf.write(str(i+1)+'\t'+ str(r_avg[0,i]) +'\t' + str(r_SE[i])+'\n')       


# In[ ]:
"""



