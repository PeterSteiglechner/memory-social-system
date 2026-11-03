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
PHASE_PARAMS = {"mh": ["logGstarmean", "logEstarmean"], "ci": ["c", "alpha"]}   # (x, y)

# %%

# ------------------------------------------------------------------ helpers

def param_labels(df, model):
    """LaTeX condition label for every row (single source of truth)."""
    if model == "mh":
        return (r"$E=" + logist(df["Estarmean"]).map("{:.2f}".format) #+ _sig(SIG_E)
                + r"$, $G=" + logist(df["Gstarmean"]).map("{:.2f}".format) #+ _sig(SIG_G)
                #+ r"$, $D=" + df["D"].map("{:.0f}".format) 
                + "$")
    return (r"$c=" + df["c"].map("{:.1f}".format)
            + r"$, $\alpha=" + df["alpha"].map("{:.1f}".format) + "$")


# ------------------------------------------------------------ theta over k
THETA = {  # key: (symbol, numerator)
    # "final":   (r"\theta_{\mathrm{final}}", r"|y_N-T|"),
    "ind":     (r"\theta_{\mathrm{ind}}",   r"\sum_n |y_n-T|"),
    "indMedian":     (r"\theta_{\mathrm{indMed}}",   r"\mathrm{median}(|y_n-T|)"),
    # "wocMean": (r"\theta_{\mathrm{WoC}}",   r"|\mathrm{mean}(y_n)-T|"),
    "crowd": (r"\theta_{\mathrm{crowd}}",   r"|\hat{y}_N-T|"),
}


def plot_theta_over_k(df, n_plot, model, cmap="tab20", baseline_col="relative_error_ind"):
    df = df.copy()
    df["k"] = df["k"].astype(float).replace(0, 0.5)       # k=0 drawn at 0.5 on the log axis
    df["parameters"] = param_labels(df, model)
    conds = df["parameters"].unique()
    palette = dict(zip(conds, sns.color_palette(cmap, len(conds))))
    gfuncs = df["gfunc"].unique()
    ticks = sorted(df["k"].unique())
    den = r"\overline{E_n}\cdot D" if model == "mh" else r"\sigma"
    name = (rf"Mayer&Heck model ($R={df.R.unique()[0]}$)" if model == "mh"
            else "Copy&Integrate model")

    for theta, (sym, num) in THETA.items():
        errcol = f"relative_error_{theta}"
        fig, axs = plt.subplots(1, len(gfuncs), figsize=(12/2.54,5/2.54),
                                sharex=True, sharey=True, squeeze=False)
        axs = axs[0]
        # fig.suptitle(f"{name}, N={n_plot}, #runs={df.seed.nunique()}")

        for ax, g in zip(axs, gfuncs):
            sub = df[df.gfunc == g]
            for mask, ls in [(sub.k <= 1, "--"), (sub.k >= 1, "-")]:
                sns.lineplot(sub[mask], x="k", y=errcol, hue="parameters", hue_order=list(conds),
                             palette=palette, ax=ax, marker="o", alpha=0.5, linestyle=ls,
                             legend=False, errorbar="sd", err_kws={"alpha": 0.05}, clip_on=False)

            # best k per condition
            m = sub.groupby(["parameters", "k"], as_index=False)[errcol].mean()
            best = m.loc[m.groupby("parameters")[errcol].idxmin()]
            ax.scatter(best["k"], best[errcol], c=best["parameters"].map(palette).tolist(),
                       marker="*", s=120, edgecolor="k", linewidth=0.5, zorder=5, clip_on=False)

            # k=0 (WoC) baseline, per condition
            base = sub[sub.k == 0.5].groupby("parameters")[baseline_col].mean()
            # for c, v in base.items():
            #     ax.axhline(v, color=palette[c], ls="--", alpha=0.6, lw=0.8)

            # ax.set_title(rf"$x_n=\,${g} of last $k$ estimates")
            ax.set_title(f"{name}, N={n_plot}", fontsize=9)
            ax.set_xscale("log")
            ax.set_ylim(bottom=0)
            ax.axvspan(0.35, 0.75, color="gainsboro", zorder=-1)
            ticks = ax.get_xticks()
            ax.set_xticks([x  for x in ax.get_xticks(minor=True) if x>=1], minor=True)
            ax.set_xticks([0.5] + [x  for x in ax.get_xticks() if x>=1])
            ax.set_xticklabels(["0"] + [int(x)  for x in ax.get_xticks() if x>=1])
            ax.set_xlim(0.35, df.k.max() + 1)
            #ax.set_xticks(ticks, ["0"] + [str(int(k)) for k in ticks[1:]])
            # ax.minorticks_off()
            ax.grid(axis="y", zorder=-2)
                

        if len(conds) < 10:
            axs[0].legend(handles=[Line2D([], [], color=palette[c], marker="o", label=c) for c in conds],
                          fontsize=7, ncol=2, handlelength=2, columnspacing=1, handletextpad=0.1)
        axs[0].set_ylabel(rf"normalised error ${sym}$")# \sim \frac{{{num}}}{{{den}}}$")
        axs[-1].text(0.53, 0.01, "Control\n"+r"$k=0$", ha="center", va="bottom", fontsize=7)
        # if "output" in theta: 
        #     axs[-1].text(1.4, base.mean() + 0.02, r"$\theta_{\rm crowd}(k=0)$", color="grey")
        fig.tight_layout()
        out = f"figs/model{'1' if model == 'ci' else '2'}_results_{theta}_N{n_plot}.png"
        fig.savefig(out, dpi=600)
        # plt.close(fig)
        print(out)




#%%
KS_PLOT = [0, 1, 3, 5, 7, 9, 15, 19, 29, 39, 49, 69, 99, 127]
n = 127

errorsMH = pd.read_csv("simData/mh_N127.csv")
errorsMH["logGstarmean"] = logist(errorsMH["Gstarmean"])
errorsMH["logEstarmean"] = logist(errorsMH["Estarmean"])
errorsMHex = pd.read_csv("simData/mh_examples_N127.csv")
errorsMHex["logGstarmean"] = logist(errorsMHex["Gstarmean"])
errorsMHex["logEstarmean"] = logist(errorsMHex["Estarmean"])
errorsCI = pd.read_csv("simData/ci_N127.csv")
errorsCIex = pd.read_csv("simData/ci_examples_N127.csv")

for d in [errorsMH, errorsMHex, errorsCI, errorsCIex]:
    d["relative_error_crowd"] = np.nan
    d.loc[d.k==0, "relative_error_crowd"] = d.loc[d.k==0, "relative_error_wocMean"]
    d.loc[d.k!=0, "relative_error_crowd"] = d.loc[d.k!=0, "relative_error_final"]
        

#%%

cmap = ["#FF3B30", "#E600A9", "#00D5E8", "#FF8C42"] 


plot_theta_over_k(errorsMHex[errorsMHex.k.isin(KS_PLOT)], n, "mh", cmap=cmap)
plot_theta_over_k(errorsCIex[errorsCIex.k.isin(KS_PLOT)], n, "ci", cmap=cmap)

# %%


# -------------------------------------------------------------- phase space
def _prep(grid):
    return grid.sort_index(axis=0).sort_index(axis=1).astype(float)


def draw_grid(ax, grid, **kw):
    grid = _prep(grid)
    return ax.pcolormesh(grid.columns.values, grid.index.values,
                         np.ma.masked_invalid(grid.values), shading="nearest", **kw)

def mean_error_grid(df, params, err, k):
    return df[df.k == k].pivot_table(index=params[1], columns=params[0], values=err, aggfunc="mean")



def condition_palette(df, model, cmap):
    labels = param_labels(df, model).unique()
    return dict(zip(labels, sns.color_palette(cmap, len(labels))))


def mark_examples(ax, ex, model, cmap, label=False):
    lab = param_labels(ex, model)
    # if model == "mh":
    # pts = pd.DataFrame({"x": logist(ex["Gstarmean"]), "y": logist(ex["Estarmean"])}).drop_duplicates()
    # for x, y in pts.values:
    #     ax.scatter(x, y, s=50, color="grey", edgecolor="white", linewidth=1.5, zorder=5,
    #                label=rf"$\mu_E={y:.2f}$, $\mu_G={x:.2f}$")
    # else:
    pal = condition_palette(ex, model, cmap)
    params = PHASE_PARAMS[model]
    for _, r in ex.assign(lab=lab).drop_duplicates("lab").iterrows():
        ax.scatter(r[params[0]], r[params[1]], s=50, marker="s", facecolor="none",
                    edgecolor=pal[r["lab"]], linewidth=1.5, zorder=5, label=r["lab"])
        if label: ax.text(r[params[0]]-0.05, r[params[1]]-0.05, r['lab'], ha="left", va="top", fontsize=6)


def phase_figure(df, ks, model, *, N=100, vmax=1,
                 examples=None, example_cmap="tab10", xlabel="", ylabel="", errorTypes = ["ind", "crowd"]):
    ks = sorted(ks)

    params = PHASE_PARAMS[model]
    k_lo, k_hi = ks[0], ks[-1]

    fig = plt.figure(figsize=(18/2.54, 13/2.54), layout="constrained")
    gs = fig.add_gridspec(3, 3, width_ratios=[1,1, 0.07], height_ratios=[0.1,1,1])
    cax = fig.add_subplot(gs[1,2])
    # gsinds = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)] if model=="mh" else [(1,0),(1,1),(2,0),(2,1)]
    gsinds = [(1,0),(1,1),(2,0),(2,1)]
    axs = [fig.add_subplot(gs[gsinds[0][0],gsinds[0][1]])]
    for i,j in gsinds[1:]:
        axs.append(fig.add_subplot(gs[i, j], sharex=axs[0], sharey=axs[0]))
    # if model=="ci":
    for n, err in enumerate(errorTypes):
        ax = fig.add_subplot(gs[0,n])
        ax.axis("off")
        inderror = df.loc[df.k==0, f"relative_error_{err}"].mean()
        ax.set_title(rf"$k=0$ (Control): $\theta_{r'{'}{err}{r'}'} = {inderror:.2f}$")

    df = df[(df.N == N) & (df.k.isin([0]+ks))]

    # --- panels b, c (shared colorbar)
    for ax, k, err in zip(axs, [k_lo, k_lo, k_hi,k_hi], errorTypes*3):
        print(k, err, ax) 
        mesh = draw_grid(ax, mean_error_grid(df, params, f"relative_error_{err}", k), cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(fr"$k={k}$, $\theta_{r'{'}{err}{r'}'}$")
    fig.colorbar(mesh, cax=cax, label=rf"mean normalised error $\theta$")

    # --- axes cosmetics
    if model == "mh":
        for ax in axs:
            ax.invert_xaxis()
    for ax in axs:
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(xlabel)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(ylabel)
        else:
            ax.tick_params(labelleft=False)
        
    for ax in axs:
        mark_examples(ax, examples, model, cmap=example_cmap, label=(ax==axs[-1]))
    # --- legend of examples
    h, l = axs[-1].get_legend_handles_labels()
    # if h:
    #     axs[-1].legend(h, l, loc="center left", frameon=False, fontsize=7, ncol=2)
    fig.suptitle((rf"Mayer&Heck ($R={df.R.unique()[0]}$)" if model == "mh" else "Copy & Integrate")
                 + " model Phase Diagram")
    return fig, axs


# %%

KS, N_EVAL, COND = [3, n], n, "Most recent"
tag = "3v127" if len(KS) == 2 else ""

# --- CI
fig, axs = phase_figure(errorsCI, KS, "ci", N=N_EVAL, examples=errorsCIex, example_cmap=cmap,
                        xlabel=r"$c$ = copy rate", ylabel=r"$\alpha$ = social signal influence", errorTypes=["indMedian", "crowd"])
print(f"figs/ErrorPhase_CI{tag}.png")
fig.savefig(f"figs/ErrorPhase_CI{tag}.png", dpi=600)

# --- MH
D_MODE = R_MODE = "fixed"
fig, axs = phase_figure(errorsMH, KS, "mh", N=N_EVAL, examples=errorsMHex, example_cmap=cmap,
                        xlabel=r"$G$ = individuality = 1 - base copy rate",
                        ylabel=r"$E$ = lack of expertise", errorTypes=["indMedian", "crowd"])
print(f"figs/ErrorPhase_MH{tag}_Dfit{D_MODE}_Rfit{R_MODE}.png")
fig.savefig(f"figs/ErrorPhase_MH{tag}_Dfit{D_MODE}_Rfit{R_MODE}.png", dpi=600)

# %%
