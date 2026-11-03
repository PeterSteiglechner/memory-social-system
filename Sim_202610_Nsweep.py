#%%
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.special import expit as logist

plt.rcParams.update({"font.size": 8})

SIG_E = SIG_G = 1e-5
PHASE_PARAMS = {"mh": ["Gstarmean", "Estarmean"], "ci": ["c", "alpha"]}   # (x, y)

#%%
from Sim_202609_models_main import n_seeds,T, gfuncs, seeds, anchoring, R, sigG, sigE, SIG_CI, simulateMH, simMH, errors_from_estimatesMH, errors_from_estimates, simulate, simCI
from scipy.special import logit
from scipy.special import expit as logist

Ns = list(np.arange(3,31,2)) + list(np.arange(31,101,10)) + list(np.arange(101,302,20))  
seeds = np.arange(n_seeds*5)
n=max(Ns)
T=0.2
cmap = ["#FF3B30", "#E600A9", "#00D5E8", "#FF8C42"] 

cs = [0.2,0.8]
alphas = [0.2,0.7]
paramsToSimCI  = [(c,a) for c in cs for a in alphas]
MHparams = [(-1,-1), (1,-1), (-1,1), (1,1)]

for model in ["ci", "mh"]:
    params = PHASE_PARAMS[model]
    
    ks = [0,3,max(Ns)-1]
    if model=="ci":
        simparams = [(c,a) for c,a in paramsToSimCI]
        errors  = simCI(n,n_seeds,Ns,SIG_CI,T,ks,simparams)
    elif model=="mh":
        simparams = [(E, G) for E,G in MHparams]
        errors = simMH(n,n_seeds,Ns,T,ks,simparams, D=3, verbose = False)
        errors["logGstarmean"] = logist(errors["Gstarmean"]).round(6)
        errors["logEstarmean"] = logist(errors["Estarmean"]).round(6)
    errors["relative_error_crowd"] = errors["relative_error_final"]
    errors.loc[errors.k==0, "relative_error_crowd"] = errors.loc[errors.k==0, "relative_error_wocMean"]


    for err in  ["crowd", "indMedian"]:
        errname = "individual median error" if err=="indMedian" else ("crowd error" if err=="crowd" else err)
        errfull = f"relative_error_{err}"
        fig, ax = plt.subplots(1,1,figsize=(16/2.54, 10/2.54))

        if model=="mh": 
            for counter, (p0,p1) in enumerate(simparams):
                ax.text(0.99, 0.99-counter*0.05, fr"${params[0][0]}={logist(p0):.2f}, {params[1][0]}={logist(p1):.2f}$", color=cmap[counter], transform=ax.transAxes, va="top", ha="right", fontsize=7)        
        else:
            for counter, (p0,p1) in enumerate(simparams):
                ax.text(0.99, 0.99-counter*0.05, fr"${params[0]}={(p0):.2f}, \{params[1]}={(p1):.2f}$", color=cmap[counter], transform=ax.transAxes, va="top", ha="right", fontsize=7)        

        for counter, (p0,p1) in enumerate(simparams):
            sns.lineplot(errors.query(f"{params[0]}=={p0} and {params[1]}=={p1}"), x="N", style="k", color=cmap[counter], y=errfull, ax=ax, legend=(counter==0), style_order=[3,max(Ns)-1, 0] )
        ax.legend(
            handles=[
                Line2D([], [], color="grey", ls="-",  label="k=3"),
                Line2D([], [], color="grey", ls="--",  label="k=N"),
                Line2D([], [], color="grey", ls=":", label="k=0"),
            ]
        )
        ax.set_xscale("log")
        ax.set_ylabel(errname)
        if err=="crowd": ax.plot(Ns, np.sqrt(1/np.array(Ns)), color="grey", )
        fig.tight_layout()
        plt.savefig(f"figs/Nsweep_{model}_{err}.png", dpi=600)
#%%
