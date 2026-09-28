#%% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st
from scipy.stats import norm
from scipy.special import erf
from scipy.special import expit as logist 

import statsmodels
import time
plt.rcParams.update({"font.size":8})

SQ2 = 2 ** 0.5


#%%


def plot_ThetaOverK(df, n_plot, model):
    conditionCol = "parameters" if model=="mh" else "c"
    df["k"] = df["k"].astype(float)
    df.loc[(df.k==0), "k"] = 0.5
    n_seeds = len(df.seed.unique())
    for theta in ["final", "ind", "agg"]:
    
        fig, axs = plt.subplots(1,2,figsize=(16/2.54,9/2.54), sharey=True, sharex=True)
        fig.suptitle(rf"Mayer&Heck model, N={n_plot}, #runs={n_seeds}, $R={R}$")
        axes = axs.flatten()
        counter = 0
        thetaLabel = (
            r"normalised error $\theta_\mathrm{final} \sim \frac{|y_N-T|}{\overline{E_n} \cdot D}$" 
            if theta=="final" else 
                (r"normalised error $\theta_\mathrm{ind} \sim \sum_n \frac{|y_n-T|}{\overline{E_n} \cdot D}$"
                if theta=="ind" else 
                r"normalised error $\theta_\mathrm{WoC} \sim \frac{|\overline{y_n}-T|}{\overline{E_n} \cdot D}$" 
                )
        )   

        for curr_gfunc in ["mean", "median"]:
            ax = axes[counter]
            sub = df.query(f"gfunc=='{curr_gfunc}'")
            if model=="mh":
                sub["parameters"] = r"$\mu_E="+ logist(sub["Emean"]).map('{:.1f}'.format) + rf"\pm{sigE:.1f}" + r"$, $\mu_G=" + logist(sub["Gmean"]).map('{:.1f}'.format) + rf"\pm{sigE:.1f}"+ r"$, $D=" + sub["D"].map('{:.0f}'.format)  + r"$"
            sns.lineplot(sub, x="k", y="relative_error_"+theta, hue=conditionCol, ax=ax, marker="o", palette="Set1", alpha=0.5, linestyle="--", legend=False, errorbar="sd", err_kws={'alpha':0.05})
            sns.lineplot(sub.loc[sub.k>=1], x="k", y="relative_error_"+theta, hue=conditionCol, ax=ax, marker="o", palette="Set1", alpha=0.5, linestyle="-", legend=ax==axs[0], errorbar="sd", err_kws={'alpha':0.05})
            # ax.set_yscale("log")
            if ax==axs[0]: ax.legend(title=None, fontsize=7)
            ax.set_title(rf"$x_n=\,${curr_gfunc} of last $k$ estimates")
            mm = sub.query(fr"k==0.5 and {conditionCol}=='{sub[conditionCol].iloc[-1]}'")["relative_error_agg"].mean()
            ax.axhline(mm, color="k", linestyle="--", alpha=0.6)
            counter+=1
        axes[0].set_ylabel(thetaLabel)

        for ax in axes:
            ax.set_ylim(0,)
            ax.set_xscale("log")
            ax.grid(axis="y", zorder=-2)
            ax.set_xticks([x  for x in ax.get_xticks(minor=True) if x>=1], minor=True)
            ax.set_xticks([0.5] + [x  for x in ax.get_xticks() if x>=1])
            ax.set_xticklabels(["0"] + [int(x)  for x in ax.get_xticks() if x>=1])
            ax.fill_betweenx(ax.get_ylim(), 0.35, 0.75, color="gainsboro", zorder=-1)
            ax.set_xlim(0.35,max(df.k.unique()))
            if ax==axs[1]: ax.text(0.79,ax.get_ylim()[1]*0.92, "Control\n"+r"$k=0$", color="grey", ha="left", va="top")
            if ax==axs[1]: ax.text(1.4, mm+0.02, r"$\theta_{\rm WoC}(k=0)$", color="grey")
        plt.tight_layout()
        plt.savefig(f"figs/model{'1' if model=='ci' else '2'}_results_{theta}_N{n_plot}.png", dpi=600)
        print(f"figs/model{'1' if model=='ci' else '2'}_results_{theta}_N{n_plot}.png")
    return 


# %%
# 
n_seeds = 1000
N_evaluate = np.array([10,20,30,40,50,75,100,150,200,250,300,400,500,600,700,800,900,1000])
n = 1001
inds = np.arange(n)
T = 0.2
memory_sizes = [0] + list(range(1,20,2))+list(range(29, min(n, 100), 2))+ [199] + [999]



# %%
# -----   RUN MAYER AND HECK
# ---------------------


rng = np.random.default_rng(0)

R = 0.4
anchoring = True
MHparams = [(-1,3,0), (-1,3,-2), (-1,3,2), (0,3,0), (-1,1,0)]
sigG, sigE = (1e-5, 1e-5)

def simulate(k, E, G, D, gfunc, z, u):
    """All seeds at once. E, G, z, u have shape (n_seeds, n)."""
    L = logist(E)
    g = np.empty((n_seeds, n))
    g[:, 0] = T + D * L[:, 0] * z[:, 0]                      # first guess: no signal
    agg = np.mean if gfunc == "mean" else np.median
    pg = logist(G)                                            # precompute
    for i in range(1, n):
        signal = agg(g[:, max(0, i - k):i], axis=1)
        p_copy = 1 - (pg[:, i] + (1 - pg[:, i]) *
                      erf((signal - T) / (D * L[:, i] * SQ2)) ** 2)
        copy = u[:, i] < p_copy
        loc = T + anchoring * L[:, i] * (signal - T)
        new = loc + R * D * L[:, i] * z[:, i]
        g[:, i] = np.where(copy, signal, new)
    return g

def errors_from(g, E, D):
    L = logist(E)
    Lbar = logist(E.mean(axis=1))                      # (n_seeds,)
    agg_l, ind_l, exp_l, fin_l = [], [], [], []

    for N in N_evaluate:
        gN = g[:, :N + 1]                              # guesses 0..N
        dev = np.abs(gN - T)
        agg_l.append(np.abs(gN.mean(axis=1) - T) / (D * Lbar))
        ind_l.append((dev / (D * L[:, :N + 1])).mean(axis=1))
        exp_l.append(dev.mean(axis=1) / (D * Lbar))
        fin_l.append(dev[:, -1] / (D * Lbar))
    # each has shape (n_seeds, len(N_evaluate))
    return tuple(np.stack(a, axis=1) for a in (agg_l, ind_l, exp_l, fin_l))

frames = []
seeds = np.arange(n_seeds)

for Gmean, D, Emean in MHparams:
    E = rng.normal(Emean, sigE, (n_seeds, n))
    G = rng.normal(Gmean, sigG, (n_seeds, n))
    for gfunc in ["mean", "median"]:
        print(gfunc, Gmean, D, Emean)
        for k in memory_sizes:
            z = rng.standard_normal((n_seeds, n))
            u = rng.random((n_seeds, n))
            if k == 0:
                g = T + D * logist(E) * z
                agg_err, e_ind, e_exp, e_fin = errors_from(g, E, D)
            else:
                g = simulate(k, E, G, D, gfunc, z, u)
                agg_err, e_ind, e_exp, e_fin = errors_from(g, E, D)

            frames.append(pd.DataFrame({
                "seed": np.repeat(seeds, len(N_evaluate)),
                "N": np.tile(N_evaluate, n_seeds),
                "n": n, "gfunc": gfunc, "D": D,
                "Emean": Emean, "Gmean": Gmean, "R": R, "k": k,
                "relative_error_agg": agg_err.ravel(),
                "relative_error_ind_indSigma": e_ind.ravel(),
                "relative_error_ind": e_exp.ravel(),
                "relative_error_final": e_fin.ravel()}))

errorsMH = pd.concat(frames, ignore_index=True)
errorsMH = errorsMH.rename(columns = {"relative_error_ind_expectedSigma":"relative_error_ind"})
errorsMH["parameters"] = r"$\mu_E="+ logist(errorsMH["Emean"]).map('{:.1f}'.format) + rf"\pm{sigE:.1f}" + r"$, $\mu_G=" + logist(errorsMH["Gmean"]).map('{:.1f}'.format) + rf"\pm{sigE:.1f}"+ r"$, $D=" + errorsMH["D"].map('{:.0f}'.format)  + r"$"
errorsMH.loc[errorsMH.k>errorsMH.N, [c for c in errorsMH.columns if "relative_error" in c]] = np.nan
# %%

#%%
ksPlot = [1,3,5,9]+[19,49,99]#+[199,999]
plot_ThetaOverK(errorsMH.loc[errorsMH.k.isin(ksPlot)], 100, "mh")
#%%
# ---------------------
# -----   Copy and Integrate by Marina, Alexandros, Pantelis et al
# ---------------------

rng = np.random.default_rng(0)

sig = 1
alpha = 0.5
cs = [0.2,0.5,0.8]

beliefs = rng.normal(T, sig, size=(n_seeds, n))   # generated once, shared by all configs

def simulate(k, c, gfunc):
    g = np.empty((n_seeds, n))
    g[:, 0] = beliefs[:, 0]
    copy = rng.random((n_seeds, n)) < c            # all coin flips in one draw
    agg = np.mean if gfunc == "mean" else np.median
    for i in range(1, n):
        signal = agg(g[:, max(0, i - k):i], axis=1)
        g[:, i] = np.where(copy[:, i], signal,
                           alpha * signal + (1 - alpha) * beliefs[:, i])
    return g

def metrics(g):
    agg_l, ind_l, fin_l = [], [], []
    for N in N_evaluate:
        gN = g[:, :N + 1]   # guesses 0..N
        dev = np.abs(gN - T)
        agg_l.append(np.abs(gN.mean(axis=1) - T) /sig)
        ind_l.append((dev / sig).mean(axis=1))
        fin_l.append(dev[:, -1] / (sig))
    return tuple(np.stack(a, axis=1) for a in (agg_l, ind_l, fin_l))

rows = []

# k = 0: independent of c and gfunc, so compute once
agg_l, ind_l, final_l = metrics(beliefs)
rows.append(pd.DataFrame({
    "seed": np.repeat(seeds, len(N_evaluate)), 
    "k": 0, 
    "N": np.tile(N_evaluate, n_seeds),
    "relative_error_agg": agg_l.ravel(), 
    "relative_error_ind": ind_l.ravel(),
    "relative_error_final": final_l.ravel(), 
    "gfunc": None, "c": None}))
gfuncs = ["mean", "median"]
for gfunc in gfuncs:
    print(gfunc)
    for c in cs:
        print(f"\t {c}")
        for k in memory_sizes[1:]:
            agg_l, ind_l, final_l = metrics(simulate(k, c, gfunc))
            rows.append(pd.DataFrame({
                "seed": np.repeat(seeds, len(N_evaluate)), 
                "k": k, "N": np.tile(N_evaluate, n_seeds),
                "relative_error_agg": agg_l.ravel(), "relative_error_ind": ind_l.ravel(),
                "relative_error_final": final_l.ravel(), "gfunc": gfunc, "c": c}))

errors_df = pd.concat(rows, ignore_index=True)
errors_df.loc[errors_df.k>errors_df.N, [c for c in errors_df.columns if "relative_error" in c]] = np.nan

#%%
k0 = errors_df[errors_df.k == 0]
k0_expanded = pd.concat(
    [k0.assign(gfunc=g, c=c) for g in gfuncs for c in cs],
    ignore_index=True,
)
errors_df = pd.concat([errors_df[errors_df.k != 0], k0_expanded], ignore_index=True)

plot_ThetaOverK(errors_df.loc[errors_df.k.isin(ksPlot)], 1000, "ci")
#%%

# %%


# %%
# GET MINIMUM

#%%
for model in ["ci", "mh"]:
    if model=="ci":
        modelTitle = "Copy & Integrate" 
        err = errors_df
        group_keys = ["gfunc", "c", "N", "seed"]
    else:
        modelTitle = "Mayer & Heck " 
        err = errorsMH
        group_keys = ["gfunc", "parameters", "N", "seed"]
    err_cols = ["relative_error_ind", "relative_error_final", "relative_error_agg"]

    # 1. average over seeds
    mean_df = (err
            .groupby(group_keys + ["k"], as_index=False)[err_cols]
            .mean())

    # 2. for each setting and each error metric, the k with the smallest mean error
    best_k = pd.DataFrame({
        col: mean_df.loc[mean_df.groupby(group_keys)[col].idxmin()]
                    .set_index(group_keys)["k"]
        for col in err_cols
    }).reset_index()

    fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(16/2.54, 9/2.54))
    ax = axs[1]
    ax.plot([],[], "-", color="k",label="w.r.t. average individual\nestimate error",)
    ax.plot([],[], "--", color="k", label="w.r.t. error of final\nestimate")
    ax.plot([],[], ":", color="k", label="w.r.t. agg WoC error")
    ax.legend(frameon=False)

    # hue= "parameters"
    hue="c" if model=="ci" else "parameters"
    for gfunc, ax in zip(["mean", "median"], axs):
        sns.lineplot(best_k.loc[best_k.gfunc==gfunc], y="relative_error_ind", x="N", hue=hue, ax=ax, palette="Set1", alpha=0.5, linestyle="-", legend=ax==axs[0], err_kws={'alpha':0.05})
        sns.lineplot(best_k.loc[best_k.gfunc==gfunc], y="relative_error_final", x="N", hue=hue, ax=ax, palette="Set1", alpha=0.5, linestyle="--", legend=False, err_kws={'alpha':0.05})
        sns.lineplot(best_k.loc[best_k.gfunc==gfunc], y="relative_error_agg", x="N", hue=hue, ax=ax, palette="Set1", alpha=0.5, linestyle=":", legend=False, err_kws={'alpha':0.05})
        ax.set_xscale("log")
        ax.grid(axis="y")
        ax.set_title(rf"$x_n=\,${gfunc} of last $k$ estimates")
        # ax.set_yscale("log")
        ax.set_ylim(-1,)
        ax.set_xlim(10,1000)
        ax.set_xlabel("number of raters $N$")
        ax.plot(np.linspace(10,1000), np.linspace(10,1000), color="grey", ls="-")
        ax.text(10,12, r"$k=N$" ,ha="left", va="bottom", rotation=30, color="grey")
    axs[0].set_ylabel("optimal $k$")
    axs[0].legend(fontsize=6, title=hue)
    axs[1].text(0.5,0.3,r"ensemble mean", transform=axs[1].transAxes, ha="center", va="bottom")
    fig.suptitle(modelTitle)
    plt.tight_layout()
    plt.savefig(f"figs/kopt_{model}.png")
# %%
