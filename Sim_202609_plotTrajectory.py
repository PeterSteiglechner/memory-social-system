#%%


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st
from scipy.stats import norm
from scipy.special import erf
from scipy.special import expit as logist
from scipy.special import logit
import matplotlib.colors as mcolors
#from matplotlib.colors import BoundaryNorm

plt.rcParams.update({"font.size":8})

#%%
errorsMH = pd.read_csv("simData/mh.csv", )
errorsMHexamples = pd.read_csv("simData/mh_examples.csv", )
sigE, sigG = (1e-5,1e-5)
errorsMH["parameters"] = r"$\mu_E="+ logist(errorsMH["Emean"]).map('{:.2f}'.format) + rf"\pm{sigE:.0f}" + r"$, $\mu_G=" + logist(errorsMH["Gmean"]).map('{:.2f}'.format) + rf"\pm{sigE:.0f}"+ r"$, $D=" + errorsMH["D"].map('{:.0f}'.format)  + r"$"
errorsMHexamples["parameters"] = r"$\mu_E="+ logist(errorsMHexamples["Emean"]).map('{:.2f}'.format) + rf"\pm{sigE:.0f}" + r"$, $\mu_G=" + logist(errorsMHexamples["Gmean"]).map('{:.2f}'.format) + rf"\pm{sigE:.0f}"+ r"$, $D=" + errorsMHexamples["D"].map('{:.0f}'.format)  + r"$"
errorsCI = pd.read_csv("simData/ci.csv", )
errorsCI["parameters"] = r"$c="+ errorsCI["c"].map('{:.1f}'.format) + r"$, $\alpha=" + errorsCI["alpha"].map('{:.1f}'.format) + r"$"
errorsCIexamples = pd.read_csv("simData/ci_examples.csv", )
errorsCIexamples["parameters"] = r"$c="+ errorsCIexamples["c"].map('{:.1f}'.format) + r"$, $\alpha=" + errorsCIexamples["alpha"].map('{:.1f}'.format) + r"$"

#%%
def plot_ThetaOverK(df, n_plot, model, cmap="tab20"):
    conditionCol = "parameters"
    gfuncs = df.gfunc.unique()
    df["k"] = df["k"].astype(float)
    df.loc[(df.k==0), "k"] = 0.5
    n_seeds = len(df.seed.unique())

    for theta in ["final", "ind", "wocMean"]:
        fig, axs = plt.subplots(1,len(gfuncs),figsize=(16/2.54,9/2.54), sharey=True, sharex=True)
        fig.suptitle(rf"{rf'Mayer&Heck model ($R={df.R.unique()[0]}$)' if model=='mh' else r'Copy&Integrate model'}, N={n_plot}, #runs={n_seeds}")
        if len(gfuncs)>1:
            axes = axs.flatten()
        else:
            axes = [axs]
        counter = 0
        thetaLabel = (
            r"normalised error $\theta_\mathrm{final} \sim \frac{|y_N-T|}{\overline{E_n} \cdot D}$" 
            if theta=="final" else 
                (r"normalised error $\theta_\mathrm{ind} \sim \sum_n \frac{|y_n-T|}{E_n \cdot D}$"
                if theta=="ind" else 
                r"normalised error $\theta_\mathrm{WoC} \sim \frac{|\overline{y_n}-T|}{\overline{E_n} \cdot D}$" 
                )
        )   
        for curr_gfunc in gfuncs:
            ax = axes[counter]
            sub = df.query(f"gfunc=='{curr_gfunc}'")
            if model=="mh":
                sub["parameters"] = r"$\mu_E="+ logist(sub["Emean"]).map('{:.2f}'.format) + rf"\pm{sigE:.0f}" + r"$, $\mu_G=" + logist(sub["Gmean"]).map('{:.2f}'.format) + rf"\pm{sigE:.0f}"+ r"$, $D=" + sub["D"].map('{:.0f}'.format)  + r"$"
            else:
                sub["parameters"] = r"$c="+ sub["c"].map('{:.1f}'.format) + r"$, $\alpha=" + sub["alpha"].map('{:.1f}'.format) + r"$"
            sns.lineplot(sub.loc[sub.k<=1], x="k", y="relative_error_"+theta, hue=conditionCol, ax=ax, marker="o", palette=cmap, alpha=0.5, linestyle="--", legend=False, errorbar="sd", err_kws={'alpha':0.05},clip_on=False)
            sns.lineplot(sub.loc[sub.k>=1], x="k", y="relative_error_"+theta, hue=conditionCol, ax=ax, marker="o", palette=cmap, alpha=0.5, linestyle="-", legend=(ax==axes[0]) & (len(sub.parameters.unique())<10), errorbar="sd", err_kws={'alpha':0.05},clip_on=False)
            if ax==axes[0]: ax.legend(title=None, fontsize=7)
            ax.set_title(rf"$x_n=\,${curr_gfunc} of last $k$ estimates")
            mm = sub.query(fr"k==0.5 and {conditionCol}=='{sub[conditionCol].iloc[-1]}'")["relative_error_wocMean"].mean()
            ax.axhline(mm, color="k", linestyle="--", alpha=0.6)
            
            errcol =  "relative_error_" +theta
            meanErr = sub.groupby(["parameters", "k"], as_index=False)[errcol].mean()
            best_k =  (
                meanErr.sort_values(errcol)
                .groupby("parameters", as_index=False)
                .first()
            )      
            # match the colors used by lineplot (hue order = order of appearance)
            conds = sub[conditionCol].unique()
            colors = cmap if isinstance(cmap, dict) else dict(zip(conds, sns.color_palette(cmap, len(conds))))
            ax.scatter(best_k["k"], best_k[errcol],
                    c=[colors[c] for c in best_k[conditionCol]],
                    marker="*", s=120, edgecolor="k", linewidth=0.5, zorder=5, clip_on=False)
            counter+=1
        axes[0].set_ylabel(thetaLabel)
        for ax in axes:
            ax.set_clip_on(False)
            ax.set_ylim(0,)
            ax.set_xscale("log")
            ax.grid(axis="y", zorder=-2)
            ax.set_xticks([x  for x in ax.get_xticks(minor=True) if x>=1], minor=True)
            ax.set_xticks([0.5] + [x  for x in ax.get_xticks() if x>=1])
            ax.set_xticklabels(["0"] + [int(x)  for x in ax.get_xticks() if x>=1])
            ax.fill_betweenx(ax.get_ylim(), 0.35, 0.75, color="gainsboro", zorder=-1)
            ax.set_xlim(0.35,max(df.k.unique()+1))
        axes[-1].text(0.53,0.01, "Control "+r"$k=0$", color="k", ha="center", va="bottom")
        axes[-1].text(1.4, mm+0.02, r"$\theta_{\rm WoC}(k=0)$", color="grey")
        fig.tight_layout()
        plt.savefig(f"figs/model{'1' if model=='ci' else '2'}_results_{theta}_N{n_plot}.png", dpi=600)
        print(f"figs/model{'1' if model=='ci' else '2'}_results_{theta}_N{n_plot}.png")
    return 

#%%

ksPlot = [0,1,3,5,7,9,15,19,29,39,49,69,99]#[1,3,5,9]+[19,49,99]#+[199,999]
# ksPlot=[0,1,3,9,19,99]
plot_ThetaOverK(errorsMHexamples.loc[errorsMHexamples.k.isin(ksPlot)], 100, "mh", cmap="Set1")

# %%
plot_ThetaOverK(errorsCIexamples.loc[errorsCIexamples.k.isin(ksPlot)], 100, "ci", cmap="tab10")

# %%
# %%


# fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(16/2.54, 9/2.54))
#     ax = axs[1]
#     ax.plot([],[], "-", color="k",label="w.r.t. average individual\nestimate error")
#     ax.plot([],[], "--", color="k", label="w.r.t. error of final\nestimate")
#     ax.plot([],[], ":", color="k", label="w.r.t. agg WoC error")
#     ax.legend(frameon=False)

#     # hue= "parameters"
#     hue="parameters" 
#     for gfunc, ax in zip(["mean", "median"], axs):
#         sns.lineplot(best_k.loc[best_k.gfunc==gfunc], y="bestk_relative_error_ind", x="N", hue=hue, ax=ax, palette="tab20", alpha=0.5, linestyle="-", legend=ax==axs[0], err_kws={'alpha':0.05}, marker="o")
#         sns.lineplot(best_k.loc[best_k.gfunc==gfunc], y="bestk_relative_error_final", x="N", hue=hue, ax=ax, palette="tab20", alpha=0.5, linestyle="--", legend=False, err_kws={'alpha':0.05})
#         sns.lineplot(best_k.loc[best_k.gfunc==gfunc], y="bestk_relative_error_agg", x="N", hue=hue, ax=ax, palette="tab20", alpha=0.5, linestyle=":", legend=False, err_kws={'alpha':0.05})
#         ax.set_xscale("log")
#         ax.grid(axis="y")
#         ax.set_title(rf"$x_n=\,${gfunc} of last $k$ estimates")
#         # ax.set_yscale("log")
#         ax.set_ylim(-1,)
#         ax.set_xlim(9,1001)
#         ax.set_xlabel("number of raters $N$")
#         ax.plot(np.linspace(10,1000,1000), np.linspace(10,1000,1000)-1, color="grey", ls="-")
#         ax.text(10,12, r"$k=N-1$" ,ha="left", va="bottom", rotation=20, color="grey")
#     axs[0].set_ylabel("optimal $k$")
#     axs[0].legend(fontsize=6, title=hue)
#     axs[1].text(0.5,0.3,r"ensemble mean", transform=axs[1].transAxes, ha="center", va="bottom")
#     fig.suptitle(modelTitle)
#     plt.tight_layout()
#     plt.savefig(f"figs/kopt_{model}.png")
# best_k[[f"bestk_{err}" for err in err_cols]].corr()


#%%

# ---------------------
# -----   PHASE SPACE
# ---------------------


#%%
def plot_phasespace(df, ax, ks, cbar=False, vmax=None, annot=False):
    cmap = mcolors.ListedColormap(["darkred", "tomato", "magenta", "cornflowerblue", "darkblue"])
    cmap = mcolors.ListedColormap(["#feebe2","#fbb4b9","#f768a1","#c51b8a","#7a0177"])
    norm = mcolors.BoundaryNorm([k-1 for k in ks]+[ks[-1]+1], cmap.N)

    df = df.astype(float).sort_index(axis=0).sort_index(axis=1)
    x, y = df.columns.values, df.index.values   # 0..1, equally spaced
    data = np.ma.masked_invalid(df.values)

    if cbar:
        mesh = ax.pcolormesh(x, y, data, shading="nearest", cmap="viridis", vmin=0, vmax=vmax)
        # mesh = ax.contour(x, y, data, cmap="viridis", vmin=0, vmax=vmax)
        ax.figure.colorbar(mesh, ax=ax)
    else:
        mesh = ax.pcolormesh(x, y, data, shading="nearest", cmap=cmap, norm=norm)
        # mesh = ax.contour(x, y, data, cmap=cmap, norm=norm, levels=[ks[0]-1]+[k+1 for k in ks])

    if annot:
        for i, yy in enumerate(y[1::4]):
            for j, xx in enumerate(x[1::4]):
                i = list(y).index(yy)
                j = list(x).index(xx)
                if not np.isnan(df.iat[i, j]):
                    ax.text(xx, yy, f"{df.iat[i, j]:.0f}", ha="center", va="center", fontsize=10)
    ax.set_xlabel(df.columns.name)
    ax.set_ylabel(df.index.name)

def plotPhaseMosaic(df, ks, model, err, scatter=True, scatter_df= None):
    params = ["logGmean", "logEmean"] if model == "mh" else ["c", "alpha"]
    minK_ind = (df.pivot_table(index=params, columns="k", values=err)
                .T.idxmin().rename("bestK").reset_index())
    minK_ind["bestK"] = pd.Categorical(minK_ind["bestK"], categories=ks)
    for k in ks:
        minK_ind = minK_ind.merge(
            df.loc[df.k == k].groupby(params)[err].mean()
            .rename(f"k{k}_mean_" + err).reset_index())

    fig, axs = plt.subplot_mosaic([["a", "a", "leg","leg2"], ["b", "b", "c", "c"]],
                                  figsize=(16 / 2.54, 14 / 2.54), height_ratios=[1,0.6])
    axs["leg2"].axis("off")
    plot_phasespace(minK_ind.pivot_table(index=params[1], columns=params[0],
                                         values="bestK", aggfunc="first"),
                    axs["a"], ks=ks, annot=True)
    for l, k in zip(["b", "c"], [3, 99]):
        axs[l].set_title(f"Mean Individual Error, k={k}")
        plot_phasespace(minK_ind.pivot_table(index=params[1], columns=params[0],
                                             values=f"k{k}_mean_" + err),
                        axs[l], ks=ks, cbar=True, vmax=1)
    # scatter: plain data coordinates, no conversion needed
    if scatter:
        paramConfigs = scatter_df.groupby(params)[params[0]].count().index
        colors = plt.get_cmap("tab20" if model=="mh" else "tab10").colors
        if model=="mh":
            for n, (G, E) in enumerate(paramConfigs):
                axs["a"].scatter(G, E, s=50, linewidth=1.5, edgecolors="white",
                                zorder=5, color="grey",
                                label=fr"$\mu_E={E:.2f}$, $\mu_G={G:.2f}$")
        else:
            for n, (c, alpha) in enumerate(paramConfigs):
                axs["a"].scatter(c, alpha, s=50, linewidth=1.5, edgecolors=colors[n],
                                zorder=5, facecolor=(0,0,0,0),
                                label=fr"$c={c:.1f}$,$\alpha={alpha:.1f}$", marker="s", )
        handles, labels = axs["a"].get_legend_handles_labels()
        axs["leg"].legend(handles, labels, loc="center", frameon=False)
    axs["leg"].axis("off")   # hide ticks, spines, background
    # axs["a"].legend(bbox_to_anchor=(1., 0.0), loc="lower center")

    for a in axs.values():
        if model=="mh":
            a.set_xlim(a.get_xlim()[1], a.get_xlim()[0])   # flip x as before
        # a.invert_yaxis()                               # ci: origin top, as with sns.heatmap

    fig.suptitle((fr"Mayer&Heck ($R={df.R.unique()[0]}$)" if model=="mh" else "Copy & Integrate") + " model Phase Diagram")
    fig.tight_layout()
    return fig, axs

#%%
ks = errorsCI.k.unique().tolist()
ks = [1,3,9,19,99]
N_evaluate = 100
fig, axs = plotPhaseMosaic(errorsCI.loc[(errorsCI.N==N_evaluate) & (errorsCI.k.isin(ks))], ks, "ci", "relative_error_ind", scatter=True, scatter_df=errorsCIexamples)

for ax in axs:
    axs[ax].set_ylabel(r"$\alpha$ = social signal influence")
    axs[ax].set_xlabel(r"$c$ = copy rate")

simoiuFit = pd.read_csv("processed_data/simoiu_fitsCI.csv")
print(simoiuFit.loc[simoiuFit.cond.isin(["Most recent", "Consensus"])].groupby(["domain_name", "cond"])[["c", "c_lo", "c_hi"] + ["alpha", "alpha_lo", "alpha_hi"]].agg("count").round(2).to_markdown())

fits = simoiuFit.loc[simoiuFit.cond.isin(["Most recent", "Consensus"])].groupby(["domain_name", "task_id", "cond"], as_index=False).first()
fits["'Most recent' outperforms 'Consensus'"] = (fits["bestCondition"] == "Most recent").astype(int)
share = (
    fits.groupby("domain_name")["'Most recent' outperforms 'Consensus'"].mean()
    .rename("fraction tasks in which 'Most recent' outperforms 'Consensus'")      
)
params = simoiuFit.groupby("domain_name")[["c", "alpha"]].mean()
params["share"] = share

norm = mcolors.Normalize(vmin=0, vmax=1)
cmapDots = plt.get_cmap("coolwarm")

sc = axs["a"].scatter(params["c"], params["alpha"], c=params["share"], s=100, linewidth=1, edgecolors="white", zorder=10,cmap=cmapDots, norm=None)
for nl, (name, row) in enumerate(params.iterrows()):
    axs["a"].text(row["c"], row["alpha"], str(nl), fontsize=6, ha="center", va="center", zorder=14)
    axs["leg2"].text(0., 1-nl/len(params), f"{nl}: {name}", fontsize=6, ha="left")
for cond, marker in zip(["Most recent", "Consensus"], ["X", "D"]):
    fitsCond = fits.loc[fits.cond==cond]
    scTasks = axs["a"].scatter(fitsCond["c"], fitsCond["alpha"], c=fitsCond["'Most recent' outperforms 'Consensus'"], s=7, linewidth=0.2, edgecolors="white", zorder=11,cmap=cmapDots, norm=None, marker=marker, alpha=0.3)
cb = plt.colorbar(sc)   # or fig.colorbar(...)
cb.set_label(r"Share of tasks in Simouiu et al where"+"\n"+r"$\theta_{ind}^{\rm Most\ recent} < \theta_{ind}^{\rm Consensus}$")
cb.set_ticks([0, 0.5, 1])
plt.savefig("figs/phase_CI"+("3v100" if len(ksPlot)==2 else '')+".png", dpi=600)

# %%
ks = errorsMH.k.unique().tolist()


errorsMH["logGmean"] = errorsMH["Gmean"].map(logist) 
errorsMH["logEmean"] = errorsMH["Emean"].map(logist) 
fig, axs =  plotPhaseMosaic(errorsMH.loc[(errorsMH.N==N_evaluate) & (errorsMH.k.isin(ks))], ks, "mh", "relative_error_ind", scatter=False, scatter_df=errorsMHexamples )
for ax in axs:
    axs[ax].set_ylabel(r"$\mathrm{logistic}(E)$ = noviceness")
    axs[ax].set_xlabel(r"$\mathrm{logistic}(G)$ = 1 - base copy rate")
# plt.savefig("figs/phase_MH"+("3v100" if len(ks)==2 else '')+".png", dpi=600)


# %%
