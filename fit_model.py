#Fit bmws model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import jax
import argparse
import re
import os
from math import log, exp, sqrt
from bmws import Observation, sim_and_fit, sim_wf
from bmws.betamix import forward, BetaMixture
from bmws.data import Dataset
from bmws.estimate import empirical_bayes, estimate, jittable_estimate, sample_paths
from bmws.sim import sim_admix
import logging

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'

rng = np.random.default_rng()

logging.disable(logging.CRITICAL)

pop_cols={  "EAS": ["#2F5233", "#B1D8B7", "#76B947"],
            "EUR": ["#BBD2FD", "#63D5F9","#2A67D6"],
            "SAM": ["#B03437", "#5885AF"],
            "SAS": ["#AE307F", "#6E2ED9"],
            "AFR": ["#FF8247", "#CD853F","#E8AD17"],
            }

################################################################
#Load data
def read_data(pop):

    pop=pop.lower()
    
    admixture_proportions=pd.read_csv("data/"+pop+"_sample_info.txt", sep="\t")
    admixture_proportions["generation"]=[int(x) for x in round(admixture_proportions["Date"]/30)]
    admixture_proportions=admixture_proportions[(admixture_proportions['Date'] <=10000)]

    #merge allele counts 
    counts=pd.read_csv("data/"+pop+"_snp_acs.raw", sep="\t")
    snps=list(counts.columns)[6:]
    data=pd.merge(admixture_proportions, counts, on="IID")

    #Parameters for data matrices
    T=max(data["generation"])+1
    N=max(data["generation"].value_counts().values)
    K=admixture_proportions.shape[1]-7
    datasets=[]
    for snp in snps:
        records = []
        for gen, count in data["generation"].value_counts().items():
            this_data=data[data["generation"]==gen]
            M=this_data.shape[0]
            for i in range(M):
                if not this_data[snp].isna().iloc[i]:
                    rec = {'t': gen}
                    rec['obs'] = (1, int(this_data[snp].values[i] / 2))
                    rec['theta'] = [this_data["k"+str(k+1)].iloc[i] for k in range(K-1)]
                    rec['theta'].append(1 - sum(rec['theta']))
                    records.append(rec)
        
        datasets.append(Dataset.from_records(records))
        
    return datasets, snps

################################################################
#Run analysis 
def run_analysis(data, alpha=1e4, beta=1, gamma=0, em_iterations=10, s=None):
    M = 100
    Ne=np.full([data.T, data.K], 1e4)
    Ne_fit=Ne
    if s is None:
        s = np.zeros([data.T, data.K])
    ab = np.ones([2, data.K]) + 1e-4
    estimate_kwargs={"alpha": alpha, "beta": beta, "gamma": gamma}

    print("Analysis")
    print("alpha=" + str(alpha) + "; beta=" + str(beta))
    for i in range(em_iterations):
        print("Iteration "+str(i))
        print("estimate_kwargs "+str(estimate_kwargs))
        ab, prior = empirical_bayes(ab0=ab, s=s, data=data, Ne=Ne, M=M)
        s = estimate(data=data, Ne=Ne_fit, prior=prior, **estimate_kwargs)
            
    return s, prior, Ne

################################################################
#Resampling by parametric bootstrapping

def resample(s, data, Ne, prior, N=10, em_iterations=10, alpha=1e4, beta=1):
    paths=sample_paths(s, Ne, data, prior, 100)
    #Now resample observations
    dataset_samples=[]
    s_samples=[]
    print("Resampling")
    for i in range(N):
        print("Resampling iteration "+str(i))
        t=data.t
        theta=data.theta
        obs=data.obs
        records = []
        Tmax=int(max(t))
        for k in range(obs.shape[0]):
            n = int(obs[k][0])
            rec = {'t': int(t[k])}
            # rec['obs'] = (n ,rng.binomial(n , float(np.sum(paths[i,:,Tmax-int(t[k])]*data.theta[k]))))
            p = rng.choice(paths[i, :, Tmax-int(t[k])], p=data.theta[k])
            rec['obs'] = (n , rng.binomial(n, p))
            rec['theta'] = data.theta[k]
            records.append(rec)
        
        dataset_samples.append(Dataset.from_records(records))
        #run inference
        s_s,prior_s,Ne_s = run_analysis(dataset_samples[i], alpha=alpha, beta=beta, em_iterations=em_iterations, s=s)
        s_samples.append(s_s)
    
    s_samples=np.stack([np.transpose(x) for x in s_samples], axis=0)
    
    return s_samples, paths

################################################################                                                                                                    
#Resampling individuals                                                                                                                                                         
def resample_individuals(s, data, Ne, prior, N=10, em_iterations=10, alpha=1e4, beta=1):
    paths=sample_paths(s, Ne, data, prior, N)
    #Now resample observations                                                                                                                                      
    dataset_samples=[]
    s_samples=[]
    print("Resampling")
    for i in range(N):
        print("Resampling iteration "+str(i))
        t=data.t
        theta=data.theta
        obs=data.obs
        records = []
        Tmax=int(max(t))

        present_obs=(data.t==0).nonzero()[0]
        nonzero_anc_obs=np.logical_and( data.obs[:,0]>0, data.t!=0).nonzero()[0]
        
        for k in range(len(present_obs)): #keep present-day individuals. 
            pik=present_obs[k]
            rec = {'t': data.t[pik]}
            rec['obs'] = (int(obs[pik][0]), int(obs[pik][1]))
            rec['theta'] = data.theta[pik]
            records.append(rec)

        pik=0 #include earlist timepoint to maintain size
        rec = {'t': data.t[pik]}
        rec['obs'] = (int(obs[pik][0]), int(obs[pik][1]))
        rec['theta'] = data.theta[pik]
        records.append(rec)
            
        for k in range(len(nonzero_anc_obs)): #resample ancient individuals
            pik=np.random.choice(nonzero_anc_obs)
            rec = {'t': data.t[pik]}
            rec['obs'] = (int(obs[pik][0]), int(obs[pik][1]))
            rec['theta'] = data.theta[pik]
            records.append(rec)
         
        dataset_samples.append(Dataset.from_records(records))
        #run inference
        s_s,prior_s,Ne_s = run_analysis(dataset_samples[i], alpha=alpha, beta=beta, em_iterations=em_iterations, s=s)
        s_samples.append(s_s)

    s_samples=np.stack([np.transpose(x) for x in s_samples], axis=0)

    return s_samples, paths


################################################################

def plot_trajectories(options, s, s_samples, paths, data):
    cols=pop_cols[options.pop]
    K=len(cols)

    high = np.quantile(s_samples, 0.05, axis=0)
    low = np.quantile(s_samples, 0.95, axis=0)
    
    fig, axs = plt.subplots(ncols=3, nrows=1,figsize=(10,3))
    for k in range(K):
        axs[0].plot(s[:, k], color=cols[k], alpha=1)
        axs[0].fill_between(range(s_samples.shape[2]), low[k], high[k], alpha=0.2, color=cols[k])

    #axs[1].plot(np.mean(s_samples, axis=0)[0], color="#2F5233")
    #axs[1].plot(np.mean(s_samples, axis=0)[1], color="#B1D8B7")
    #axs[1].plot(np.mean(s_samples, axis=0)[2], color="#76B947")

    high = np.max(paths,  axis=0)
    low = np.min(paths,  axis=0)
    for k in range(K):
        axs[1].plot(np.mean(paths, axis=0)[k][::-1], color=cols[k])
        axs[1].fill_between(range(paths.shape[2]), low[k][::-1], high[k][::-1], alpha=0.2, color=cols[k])

    a,b=[int(y) for x,y in zip(data.obs, data.t) if x[0]>0],[int(x[1]) for x,y in zip(data.obs, data.t) if x[0]>0]
    sns.regplot(x=a, y=b, logistic=True, ax=axs[2], color="black")

    axs[0].set(xlabel="Generations before present", ylabel="Selection coefficient")
    axs[1].set(xlabel="Generations before present", ylabel="Allele frequency")
    axs[2].set(xlabel="Generations before present", ylabel="Observations")

    #fig.suptitle(pop+": "+snp, y=1.1)

    plt.tight_layout()
    
    outdir=options.outdir
    if outdir and outdir[-1]!="/":
        outdir=outdir+"/"
    
    filename=outdir+options.pop+"_"+re.sub( r'_.+$' , "", options.snp)+".pdf"
    fig.savefig(filename)
    
    return

################################################################

def parse_options():
    """
    argparse
    """
    parser=argparse.ArgumentParser()

    parser.add_argument('-p', '--pop', type=str, default="", help=
                        "population code")
    parser.add_argument('-s', '--snp', type=str, default="", help=
                        "Which SNP")
    parser.add_argument('-e', '--em', type=int, default=10, help=
                        "Number of EM iterations")
    parser.add_argument('-f', '--fm', type=int, default=2, help=
                        "Number of resampling EM iterations")
    parser.add_argument('-N', '--N', type=int, default=100, help=
                        "Number of resamples")
    parser.add_argument('-o', '--outdir', type=str, default="", help=
                        "Where to put plots")
    parser.add_argument('-a', '--loga', type=float, default=4.0, help=
                        "log10 alpha")
    parser.add_argument('-b', '--logb', type=float, default=1.0, help=
                        "log10 beta")
    parser.add_argument('-r', '--resample', type=str, default="parametric",
                        choices=["parametric", "individual"], help=
                        "resampling method")

    return parser.parse_args()

################################################################

def main(options):
    alpha = pow(10,options.loga)
    beta = pow(10,options.logb)

    datasets, snps=read_data(options.pop)
    data=datasets[snps.index(options.snp)]
    kw = dict(data=data, alpha=alpha, beta=beta, em_iterations=options.em)
    s,prior,Ne=run_analysis(**kw)
    if options.resample=="parametric":
        f = resample
    elif options.resample=="individual":
        f = resample_individuals
    kw = dict(s=s, data=data, Ne=Ne, prior=prior, N=options.N, em_iterations=options.fm, alpha=alpha, beta=beta)
    s = kw['s']
    data = kw['data']
    s_samples, paths = f(**kw)
    plot_trajectories(options, s, s_samples, paths, data)
    return

################################################################

if __name__=="__main__":
    options=parse_options()
    main(options)
