#!/usr/bin/env python
# coding: utf-8

# In[39]:


from bmws.data import Dataset
from bmws.sim import sim_admix
import bmws.infer
import numpy as np
import scipy

D = 3
T = 300
rng = np.random.default_rng(1)
alpha = np.ones(D)
recs = [dict(t=rng.integers(0, T), theta=rng.dirichlet(alpha), obs=(1, 1))
        for i in range(T)]
recs.extend([dict(t=0, theta=rng.dirichlet(alpha), obs=(1, 1))
        for i in range(T)])
data = Dataset.from_records(recs)

s = np.zeros([data.T, D])
s[:, 0] = 0.05
# s[:, 2] = -0.01
f0 = np.full([D], 0.01)

data, afs = sim_admix(
    mdl=dict(s=s, f0=f0, Ne=1e4),
    data=data,
    seed=1,
)

print(data.theta)
print(afs)
dl = -np.diff(scipy.special.logit(afs), axis=0)
s_hat = np.expm1(dl) * 2
xp = afs[:-1]
x = afs[1:]
s_hat1 = 2 * (xp - x) / x / (1 - xp)

# In[48]:

import bmws.betamix
import importlib
import jax
importlib.reload(bmws.betamix) 
sln = bmws.infer.Selection.default(T, D)
a = b = np.ones(D)
ab = np.array([a, b])
bmws.infer.gibbs(sln, data, niter=100, alpha=10., M=5000, N_E=10_000, mean_paths=None)
breakpoint()


# In[47]:


get_ipython().run_line_magic('debug', '')

