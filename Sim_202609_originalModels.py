import numpy as np
from scipy.special import expit as logist
from scipy.special import erf
import scipy.stats as st
import pandas  as  pd

# Slow Models

# Variables: 
# E \in R represents noviceness (or inverse expertise), where E=0 is the average expertise level
# G \in R represents tendency to change. When signal==truth, how likely is it that you change something? G=0 means, p to adjust is 50%. G=-1 means p(y==truth)=0.268
# D \in R is difficulty 
# 0<R<=1 is reduction factor of the 'plausibility frame' for new signal creation (compared with acceptance)
# p is propbability to adjust a 


# def p_original(signal, truth, G, D, E): 
#     return logist(G) + (1-logist(G)) * 4 * (norm.cdf((signal - truth)/(D*logist(E)))- 0.5)**2 

# logist = lambda x: (1/(1+np.exp(-x)))

from scipy.special import expit as logist


def p(signal, truth, G, D, E): 
    return logist(G) + (1-logist(G)) * (erf((signal - truth)/(D*logist(E)*(2**0.5))))**2 

def new_signal(signal, truth, D, E, R, anchoring=True): 
    if np.isnan(signal):
        loc=truth
        scale=D*logist(E)
    else:
        loc = truth + int(anchoring) * (logist(E)) * (signal - truth)
        scale = R*D*logist(E)
    dist = st.norm(loc=loc, scale=scale)
    return dist.rvs()

agg = lambda x, which: np.median(x) if which=="median" else np.mean(x) # performance aggregtaion

#%%

def slowMH(n, N_evaluate, k, MHparams, n_seeds, T=0.2, R=0.4,anchoring=True, sigE=1e-5, sigG=1e-5):
    #MHparams = (Gmean, D, Emean)
    inds = np.arange(n)
    errorsMH = []
    for gfunc in ["mean", "median"]:
        print(f"{gfunc}")
        sigG, sigE = (1e-5, 1e-5)
        for Gmean, D, Emean in MHparams:
            for seed in range(n_seeds):
                np.random.seed(seed)
                E = st.norm(Emean,sigE).rvs(n) # [0]*n
                G = st.norm(Gmean,sigG).rvs(n)
                if seed==0: print(f"Gmean: {Gmean} (and {logist(np.mean(G)):.3f}), Emean={Emean} (and {logist(np.mean(E)):.3f}), D={D}, sigG={sigG}, sigE={sigE}")
                guesses = []
                copier=0
                for i in inds:
                    last_k_guesses = [] if k==0 else (guesses if k>=len(guesses) else guesses[-k:])
                    signal = np.nan if len(last_k_guesses)==0 else agg(last_k_guesses, gfunc)
                    if not np.isnan(signal) and k>0:
                        p_ik = 1 - p(signal, T,  G=G[i], D=D, E=E[i]) # probability copy
                        copy = int(np.random.random()<p_ik) 
                        copier+= copy
                    if i==inds[0] or k==0: 
                        guess = new_signal(np.nan, T, D=D, E=E[i], R=1, anchoring=False)
                    else:
                        guess = signal if copy else new_signal(signal, T, D=D, E=E[i], R=R, anchoring=anchoring)
                    guesses.append(guess)
                    if i in N_evaluate:
                        relerror_agg = abs(agg(guesses, "mean") - T)/(D*logist(np.mean(E)))
                        relerror_ind_indSigma = np.mean([abs(g-T)/(D*logist(E[i])) for i,g in enumerate(guesses)])
                        relerror_ind_expectedSigma = np.mean([abs(g-T) for g in guesses])/(D*logist(np.mean(E)))
                        relerror_final = abs(guesses[-1]-T)/(D*logist(np.mean(E)))
                        errorsMH.append([seed, i, gfunc, D, Emean, Gmean, R, k, relerror_agg, relerror_ind_expectedSigma, relerror_final, relerror_ind_indSigma])
                                
                    # print(f"nr of copy-decisions (last seed) with k={k}: {copier} of {len(inds)-1} decisions")

    return pd.DataFrame(errorsMH, columns = ["seed", "N", "gfunc", "D", "E", "G", "R", "k", "relative_error_agg",  "relative_error_ind", "relative_error_final", "relative_error_ind_indSigma",])


#%%


def slowCopyIntegrate(n, N_evaluate, k, c, alpha, sig, n_seeds, T=0.2):
    inds = np.arange(n)
    errors = []
    for gfunc in ["mean", "median"]:
            print(gfunc)
            for seed in range(n_seeds):
                np.random.seed(seed)
                private_beliefs = st.norm(loc=T, scale=sig).rvs(n)
                guesses = []
                for i in inds:
                    last_k_guesses = [] if k==0 else (guesses if k>=len(guesses) else guesses[-k:])
                    signal = np.nan if len(last_k_guesses)==0 else agg(last_k_guesses, gfunc)
                    if not np.isnan(signal) and k>0: 
                        copy = bool(np.random.random()<c) 
                    if i==inds[0] or k==0: 
                        guess = private_beliefs[i]
                    else:
                        guess = signal if copy else (alpha * signal + (1-alpha) * private_beliefs[i])
                    guesses.append(guess)

                    if i in N_evaluate:
                        relerror_agg = abs(agg(guesses, "mean") - T)/(sig)
                        relerror_ind = np.mean([abs(g-T) for g in guesses])/sig
                        relerror_final = abs(guesses[-1]-T)/sig
                        errors.append([seed, i, gfunc, c, alpha, sig, k, relerror_agg, relerror_ind, relerror_final])
    return pd.DataFrame(errors, columns = ["seed", "N", "gfunc", "c", "alpha", "sig", "k", "relative_error_agg",  "relative_error_ind", "relative_error_final"])



def getK(df, N_evaluate, params, paramColumn):
    minKs = []
    for gfunc in ["mean", "median"]:
        for N in N_evaluate:
            sub = df.query(f"gfunc=='{gfunc}' and N=={N}")[["k", paramColumn, "seed", "relative_error_ind", "relative_error_final"]]
            for p in params:
                for s in sub.seed.unique():
                    kmin = sub["k"].iloc[sub.groupby(paramColumn)["relative_error_ind"].mean().argmin()]
                    kminLast = sub["k"].iloc[sub.groupby(paramColumn)["relative_error_final"].mean().argmin()]
                    minKs.append([gfunc, p, N, s, kmin, kminLast])
    best_k = pd.DataFrame(minKs, columns=["gfunc", paramColumn, "N", "seed", "kmin", "kmin_finalGuess"])
#%%
