# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import jax
from math import log, exp, sqrt


from bmws import Observation, sim_and_fit, sim_wf
from bmws.betamix import forward, BetaMixture
from bmws.data import Dataset
from bmws.estimate import empirical_bayes, estimate, jittable_estimate, _beta_pdf
from bmws.sim import sim_admix
rng = np.random.default_rng()


# +
#This block is just reading in the data, sorry it's a bit gross 

#Load EAS data
admixture_proportions=pd.read_csv("admixture_proportions.txt", sep="\t")
admixture_proportions["generation"]=[int(x) for x in round(admixture_proportions["Date"]/30)]
admixture_proportions=admixture_proportions[(admixture_proportions['Date'] <=10000)]

#merge allele counts 
counts=pd.read_csv("snp_acs.raw", sep="\t")
data=pd.merge(admixture_proportions, counts, on="IID")

#Spread present-day samples randomly over last 10 generations, for computational efficiency. 
for i in range(data.shape[0]):
    if data.iloc[i, data.columns.get_loc('generation')]==0:
        data.iloc[i, data.columns.get_loc('generation')]=rng.choice(10)

#Parameters for data matrices
T=max(data["generation"])+1
N=max(data["generation"].value_counts().values)

#Which SNP to look at
snp="rs7925299_C(/G)"

#Create data matrices
obs = np.zeros([T, N, 2], dtype=int)
samples = np.zeros([T, N], dtype=int)
#thetas = np.zeros([T, N, 3], dtype=float)
thetas = rng.dirichlet(np.ones(3), [T, N])
# -

# # New data format
#
# Create a list of "records" and then call `data = Dataset.from_records()`. Each record is of the form:
#
#     rec = {
#         't': t,  # time of observation (gens before present)
#         'theta': theta  # [K] admixture proportions,
#         'obs': (n, d)   # tuple: n = # of observed alleles, d = # of derived alleles.
#     }
#     
# (For diploid data, $n=2$, while for pseuodhaploid data $n=1$)

# +
#Fill in data matrices (in simplest way possible) - generation, N
records = []

for gen, count in data["generation"].value_counts().items():
    this_data=data[data["generation"]==gen]
    M=this_data.shape[0]
    for i in range(M):
        if not this_data[snp].isna().iloc[i]:
            rec = {'t': gen}
            rec['obs'] = (1, int(this_data[snp].values[i] / 2))
            rec['theta'] = [this_data["North"].iloc[i], 
                            this_data["South"].iloc[i]]
            rec['theta'].append(1 - sum(rec['theta']))
            records.append(rec)
# -

data = Dataset.from_records(records)

# +
#Run analysis - no longer fails!
em_iterations=1
M=100
Ne=np.full([data.T, data.K], 1e4)
Ne_fit=Ne
s = np.zeros([data.T, data.K])
ab = np.ones([2, data.K]) + 1e-4
estimate_kwargs={"lam": 1e4, "gamma": 0.0}

with jax.debug_nans(True):
    for i in range(em_iterations):
        logger.info("EM iteration %d", i)
        ab, prior = empirical_bayes(ab0=ab, s=s, data=data, Ne=Ne, M=M)
        logger.info("ab: %s", ab)
        s = estimate(data=data, Ne=Ne_fit, prior=prior, **estimate_kwargs)
        logger.info("s: %s", s)

betas, _ = forward(s, Ne, data, prior)
# -

plt.plot(s[:, 0], color="tab:blue", alpha=1)
plt.plot(s[:, 1], color="tab:orange", alpha=1)
plt.plot(s[:, 2], color="tab:green", alpha=1)

# +
#Replace observations with random alleles with frequency 50%
#This runs fine. 

obs2 = np.zeros([T, N, 2], dtype=int)
for t in range(T):
    for n in range(N):
        a = obs[t, n, 0]
        d = rng.binomial(a, 0.5)
        obs2[t, n] = [a, d]
data, nzi = Dataset(thetas=thetas, obs=obs2).resort()

em_iterations=1
M=100
Ne=np.zeros([T - 1, data.K]) + 10000
Ne_fit=Ne
s = np.zeros([T - 1, data.K])
ab = np.ones([2, data.K]) + 1e-4
estimate_kwargs={"lam": 1e4, "gamma": 0.0}

for i in range(em_iterations):
    logger.info("EM iteration %d", i)
    ab, prior = empirical_bayes(ab0=ab, s=s, data=data, nzi=nzi, Ne=Ne, M=M)
    logger.info("ab: %s", ab)
    s = estimate(data=data, Ne=Ne_fit, prior=prior, nzi=nzi, **estimate_kwargs)
    logger.info("s: %s", s)

betas, _ = forward(s, Ne, data, nzi, prior)

# -


