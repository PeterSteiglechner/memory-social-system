#%% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st
from scipy.stats import norm
from scipy.special import erf
import statsmodels
import time
plt.rcParams.update({"font.size":8})

#%%
agg = lambda x, which: np.median(x) if which=="median" else np.mean(x) # performance aggregtaion


# %%
# Mayer and Heck 2025
# 
# Variables: 
# E \in R represents noviceness (or inverse expertise), where E=0 is the average expertise level
# G \in R represents tendency to change. When signal==truth, how likely is it that you change something? G=0 means, p to adjust is 50%. G=-1 means p(y==truth)=0.268
# D \in R is difficulty 
# 0<R<=1 is reduction factor of the 'plausibility frame' for new signal creation (compared with acceptance)
# p is propbability to adjust a 

#%%
logist = lambda x: (1/(1+np.exp(-x)))
# def p_original(signal, truth, G, D, E): 
#     return logist(G) + (1-logist(G)) * 4 * (norm.cdf((signal - truth)/(D*logist(E)))- 0.5)**2 
def p(signal, truth, G, D, E): 
    return logist(G) + (1-logist(G)) * (erf((signal - truth)/(D*logist(E)*(2**0.5))))**2 
def new_signal(signal, truth, D, E, R, anchoring=True): 
    if np.isnan(signal):
        loc=truth
        scale=D*logist(E)
    else:
        loc = truth + int(anchoring) * (logist(E)) * (signal - truth)
        scale = R*D*logist(E)
    dist = norm(loc=loc, scale=scale)
    return dist.rvs()

#%%
T = 0.2
plt.figure(figsize=(12/2.54, 5/2.54))
ax = plt.axes()
x = np.linspace(-1,1)
for G, D, E in [(-1,3,0), (-1,3,-2), (-1,3,2), (0,3,0), (-1,1,0)]:
    l, = ax.plot(x, 1-p(x, T,  G=G, D=D, E=E), label=f"E={logist(E):.1f}, G={logist(G):.1f}, D={D}", alpha=0.5)
    ax.text(1.05, 1-p(1, T,  G=G, D=D, E=E), f"E={logist(E):.1f}, G={logist(G):.1f}, D={D}" , ha="left", color=l.get_color(), )
ax.set_xlim(-1,1)
ax.set_yticks(np.arange(0,0.81,0.1))
ax.set_xticks(np.arange(-1,1.01,0.5))
ax.grid(axis="y")
#plt.legend(loc="lower left")
ax.set_ylabel("probability to copy\n"+r"$p(x_n|\text{params}\ E,\, D,\, G)$")
ax.set_xlabel(r"social signal $x_n$")
ax.axvline(T, color= "grey")
ax.text(T+0.03, 0.2, f"T={T}")
plt.tight_layout()
plt.savefig(f"figs/pcopy_mayerheck.png", dpi=600)


# %%
# -----   RUN MAYER AND HECK
# ---------------------
n_seeds = 1000
n_per_chain = 100
inds = range(n_per_chain)
T = 0.2
R = 0.4
anchoring = True
memory_sizes = [0] + list(range(1,20,2))+[49,99]

MHparams = [(-1,3,0), (-1,3,-2), (-1,3,2), (0,3,0), (-1,1,0)]
errorsMH = []
copyDict = {}
for ffunc, gfunc in zip(["mean", "median"], ["mean", "median"]):
    print(f"{ffunc}, {gfunc}")
    sigG, sigE = (1e-5, 1e-5)
    for Gmean, D, Emean in MHparams:
        print(f"Gmean: {Gmean} (and {logist(np.mean(G)):.3f}), Emean={Emean} (and {logist(np.mean(E)):.3f}), D={D}, sigG={sigG}, sigE={sigE}")
        for seed in range(n_seeds):
            np.random.seed(seed)
            E = st.norm(Emean,sigE).rvs(n_per_chain) # [0]*n_per_chain
            G = st.norm(Gmean,sigG).rvs(n_per_chain)
            for k in memory_sizes:
                guesses = []
                copier=0
                for i in inds:
                    last_k_guesses = guesses if k>=len(guesses) else guesses[-k:]
                    signal = np.nan if len(last_k_guesses)==0 else agg(last_k_guesses, gfunc)
                    if not np.isnan(signal):
                        p_ik = 1 - p(signal, T,  G=G[i], D=D, E=E[i]) # probability copy
                        copy = int(np.random.random()<p_ik) 
                    else:
                            copy = False
                    guess = signal if copy else new_signal(signal, T, D=D, E=E[i], R=R, anchoring=anchoring)
                    guesses.append(guess)
                    copier+= copy
                copyDict[(Gmean, D, Emean, k, seed)] = copier
                relerror = abs(agg(guesses, ffunc) - T)/(D*logist(np.mean(E)))
                errorsMH.append([seed, gfunc, ffunc, D, Emean, Gmean, R, k, relerror])
        # print(f"nr of copy-decisions (last seed) with k={k}: {copier} of {len(inds)-1} decisions")

# %%
errorsMH_df = pd.DataFrame(errorsMH, columns = ["seed", "gfunc", "ffunc", "D", "E", "G", "R", "k", "relative_error"])
# errorsMH_df = errorsMH_df.loc[errorsMH_df.k.isin([0]+list(range(1,101,2)))]
errorsMH_df["k"] = errorsMH_df["k"].astype(float)
errorsMH_df.loc[errorsMH_df.k==0, "k"] = 0.5
fig, axs = plt.subplots(1,2,figsize=(16/2.54,9/2.54), sharey=True, sharex=True)
fig.suptitle(rf"Mayer&Heck model, N={n_per_chain}, #runs={n_seeds}, $R={R}$")
axes = axs.flatten()
counter = 0
for curr_ffunc, curr_gfunc in zip(["mean", "median"], ["mean", "median"]):
    ax = axes[counter]
    sub = errorsMH_df.query(f"ffunc=='{curr_ffunc}' and gfunc=='{curr_gfunc}'")
    sub["parameters"] = r"$\mu_E="+ logist(sub["E"]).map('{:.1f}'.format)+rf"\pm{sigE:.1f}"+ r"$, $\mu_G=" + logist(sub["G"]).map('{:.1f}'.format) +rf"\pm{sigE:.1f}"+ r"$, $D=" + sub["D"].map('{:.0f}'.format)  + r"$"
    # sub["parameters"] = logist(sub["E"]).map('{:.1f}'.format)+"_" + logist(sub["G"]).map('{:.1f}'.format) +"_"+ sub["D"].map('{:.0f}'.format)  
    sns.lineplot(sub, x="k", y="relative_error", hue="parameters", ax=ax, marker="o", palette="Set1", alpha=0.5, legend=ax==axs[0], errorbar="sd")
    # ax.set_yscale("log")
    if ax==axs[0]: ax.legend(title=None, fontsize=7)
    ax.set_title(rf"f={curr_ffunc}; g={curr_gfunc}")
    # for nD, params in enumerate(sub["parameters"].unique()):
    #     mm = sub.query(fr"parameters=='{params}'")["relative_error"].mean()
    #     ss = sub.query(fr"parameters=='{params}'")["relative_error"].describe(percentiles=[0.025,0.975]).loc[["2.5%", "97.5%"]].values
    #     ax.axhline(ss[0], color=plt.get_cmap("Set1").colors[nD], linestyle="--", alpha=0.2)
    #     ax.axhline(ss[1], color=plt.get_cmap("Set1").colors[nD], linestyle="--", alpha=0.2)
    #     ax.axhline(mm, color=plt.get_cmap("Set1").colors[nD], linestyle="--")
    counter+=1

for ax in axes:
    ax.set_ylim(0,)
    ax.set_xscale("log")
    ax.grid(axis="y", zorder=-2)
    ax.set_xticks([x  for x in ax.get_xticks(minor=True) if x>=1], minor=True)
    ax.set_xticks([0.5] + [x  for x in ax.get_xticks() if x>=1])
    ax.set_xticklabels(["0"] + [int(x)  for x in ax.get_xticks() if x>=1])
    ax.fill_betweenx(ax.get_ylim(), 0.4,0.75, color="gainsboro", zorder=-1)
    if ax==axs[1]: ax.text(0.79,1., "Control\n"+r"$k=0$", color="grey", ha="left")
plt.tight_layout()
plt.savefig("figs/model2_results.png", dpi=600)

#%%

# ---------------------
# -----   Marina, Alexandros, Pantelis et al
# ---------------------

n_seeds = 1000
n_per_chain = 100
inds = range(n_per_chain)
sig = 1
alpha = 0.5
T = 0.2
memory_sizes = [0] + list(range(1,10, 2))+list(range(19,100,10))

errors = []
errorsindep = []
for ffunc, gfunc in zip(["mean", "median"], ["mean", "median"]):
        print(ffunc, gfunc)
        for c in [0.2,0.5, 0.8]:
            for seed in range(n_seeds):
                np.random.seed(seed)
                private_beliefs = st.norm(loc=T, scale=sig).rvs(n_per_chain)
                for k in memory_sizes:
                    guesses = []
                    for i in inds:
                        last_k_guesses = guesses if k>=len(guesses) else guesses[-k:] 
                        signal = np.nan if len(last_k_guesses)==0 else agg(last_k_guesses, gfunc)
                        if not np.isnan(signal): 
                            copy = bool(np.random.random()<c) 
                        else:
                            copy = False
                        if i==inds[0] or k==0: 
                            guess = private_beliefs[i]
                        else:
                            guess = signal if copy else (alpha * signal + (1-alpha) * private_beliefs[i])
                        guesses.append(guess)
                    relerror = abs(agg(guesses, ffunc) - T)/sig
                    errors.append([seed, gfunc, ffunc, c, alpha, sig, k, relerror])

# %%
errors_df = pd.DataFrame(errors, columns = ["seed", "gfunc", "ffunc", "c", "alpha", "sig", "k", "relative_error"])
errors_df["k"] = errors_df["k"].astype(float)
errors_df.loc[errors_df.k==0, "k"] = 0.5

fig, axs = plt.subplots(1,2,figsize=(16/2.54,9/2.54), sharey=True, sharex=True)
fig.suptitle(rf"Copy-vs-integrate model, N={n_per_chain}, #runs={n_seeds}, $\alpha={alpha}$, $\sigma={sig}$")
axes = axs.flatten()
counter = 0
for curr_ffunc, curr_gfunc in zip(["mean", "median"], ["mean", "median"]):
        ax = axes[counter]
        sub = errors_df.query(f"ffunc=='{curr_ffunc}' and gfunc=='{curr_gfunc}'")
        sns.lineplot(sub, x="k", y="relative_error", hue="c", ax=ax, marker="o", palette="Set1", alpha=0.5, errorbar="sd", legend=(ax==axs[0]))
        ax.set_title(rf"f={curr_ffunc}; g={curr_gfunc}")
        counter+=1
for ax in axes:
    ax.set_ylim(0,)
    ax.set_xscale("log")
    ax.grid(axis="y", zorder=-2)
    ax.set_xticks([x  for x in ax.get_xticks(minor=True) if x>=1], minor=True)
    ax.set_xticks([0.5] + [x  for x in ax.get_xticks() if x>=1])
    ax.set_xticklabels(["0"] + [int(x)  for x in ax.get_xticks() if x>=1])
    ax.fill_betweenx(ax.get_ylim(), 0.4,0.75, color="gainsboro", zorder=-1)
    if ax==axs[1]: ax.text(0.79,0.7, "Control\n"+r"$k=0$", color="grey", ha="left")
plt.tight_layout()

plt.savefig("figs/model1_results.png", dpi=600)
# %%
