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

#%%


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
        

# errorsCI = errorsCI.query("c>0 and c<1 and alpha>0 and alpha<1")
# errorsMH = errorsMH.query("Estarmean>-4 and Estarmean<4 and Gstarmean>-4 and Gstarmean<4")

#%%

# ------------------------------------------------------------------ helpers
# def _sig(s):
#     return rf"\pm10^{{{int(round(np.log10(s)))}}}"


def param_labels(df, model):
    """LaTeX condition label for every row (single source of truth)."""
    if model == "mh":
        return (r"$E=" + logist(df["Estarmean"]).map("{:.2f}".format) #+ _sig(SIG_E)
                + r"$, $G=" + logist(df["Gstarmean"]).map("{:.2f}".format) #+ _sig(SIG_G)
                #+ r"$, $D=" + df["D"].map("{:.0f}".format) 
                + "$")
    return (r"$c=" + df["c"].map("{:.1f}".format)
            + r"$, $\alpha=" + df["alpha"].map("{:.1f}".format) + "$")


def condition_palette(df, model, cmap):
    labels = param_labels(df, model).unique()
    return dict(zip(labels, sns.color_palette(cmap, len(labels))))


# -------------------------------------------------------------- phase space
def _prep(grid):
    return grid.sort_index(axis=0).sort_index(axis=1).astype(float)


def draw_grid(ax, grid, **kw):
    grid = _prep(grid)
    return ax.pcolormesh(grid.columns.values, grid.index.values,
                         np.ma.masked_invalid(grid.values), shading="nearest", **kw)


def annotate_grid(ax, grid, step=4, fontsize=8):
    grid = _prep(grid)
    for i in range(1, len(grid.index), step):
        for j in range(1, len(grid.columns), step):
            v = grid.iat[i, j]
            if np.isfinite(v):
                ax.text(grid.columns[j], grid.index[i], f"{v:.0f}",
                        ha="center", va="center", fontsize=fontsize)


def win_prob(df, params, err, k_a, k_b):
    """P(err[k_a] <= err[k_b]) over seeds; returns grid (index=y, columns=x)."""
    if df.duplicated(params + ["seed", "k"]).any():
        warnings.warn("Several rows per (params, seed, k); using 'first'.")
    wide = df.pivot_table(index=params + ["seed"], columns="k", values=err, aggfunc="first")
    wide = wide[[k_a, k_b]].dropna()
    return (wide[k_a] < wide[k_b]).groupby(level=params).mean().unstack(params[0])

def errDiff(df, params, err, k_a, k_b):
    """P(err[k_a] <= err[k_b]) over seeds; returns grid (index=y, columns=x)."""
    if df.duplicated(params + ["seed", "k"]).any():
        warnings.warn("Several rows per (params, seed, k); using 'first'.")
    wide = df.pivot_table(index=params + ["seed"], columns="k", values=err, aggfunc="first")
    wide = wide[[k_a, k_b]].dropna()
    return (wide[k_a] - wide[k_b]).groupby(level=params).mean().unstack(params[0])


def best_k_grid(df, params, err, ks):
    m = df.pivot_table(index=params, columns="k", values=err, aggfunc="mean")[ks]
    return m.idxmin(axis=1).unstack(params[0])


def mean_error_grid(df, params, err, k):
    return df[df.k == k].pivot_table(index=params[1], columns=params[0], values=err, aggfunc="mean")


def bestk_style(ks):                       # assumes sorted ks with gaps >= 2
    cmap = mcolors.ListedColormap(sns.color_palette("RdPu_r", len(ks)))
    norm = mcolors.BoundaryNorm([k - 1 for k in ks] + [ks[-1] + 1], cmap.N)
    return cmap, norm



def domain_fits(csv, raw_cols, phase_cols, condition="Most recent", other="Consensus", errSi="agg"):
    """Per-domain mean fitted params + share of tasks where `condition` wins."""
    f = pd.read_csv(csv)
    f = f[f.cond.isin([condition, other])]
    f["crowd_error"] = np.nan
    f.loc[f.cond=="Control", "crowd_error"] = f.loc[f.cond=="Control", "agg_error"]
    f.loc[f.cond!="Control", "crowd_error"] = f.loc[f.cond!="Control", "final_error"]
    best = f.pivot_table(index="task_id", columns="cond", values=errSi, aggfunc="first")[[condition, other]].idxmin(axis="columns").rename("bestConditionof2")
    f = f.join(best, on=["task_id"])
    tasks = f.groupby(["domain_name", "task_id", "cond"], as_index=False).first()
    share = (tasks["bestConditionof2"] == condition).groupby(tasks["domain_name"]).mean()
    out = f[f.cond == condition].groupby("domain_name")[raw_cols].mean()
    out.columns = phase_cols
    out["share"] = share
    return out


def mark_examples(ax, ex, model, cmap):
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


def phase_figure(df, ks, model, *, N=100, err="relative_error_ind", vmax=1,
                 examples=None, example_cmap="tab10", fits=None,
                 condition="Most recent", xlabel="", ylabel=""):
    #ks = sorted(ks)
    params = PHASE_PARAMS[model]
    k_lo, k_hi = ks[0], ks[-1]
    df = df[(df.N == N) & df.k.isin(ks)]
    
    # Layout: a spans 2 columns / 2x height; b and c are 1 column each, half the height
    # -> same aspect ratio as a, but smaller and equal to each other.
    fig = plt.figure(figsize=(16/2.54, 7/2.54), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.07, 0.2, 0.9], height_ratios=[1])
    ax_a, cax_a = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0,1])
    ax_dom = fig.add_subplot(gs[0,3])
    # ax_b = fig.add_subplot(gs[1, 0])
    # ax_c = fig.add_subplot(gs[1, 1], sharex=ax_b, sharey=ax_b)
    # cax_bc = fig.add_subplot(gs[1, 2])

    # --- panel a
    if len(ks) == 2:
        draw_grid(ax_a, win_prob(df, params, err, k_lo, k_hi), cmap="coolwarm", vmin=0, vmax=1)
    else:
        cmap, norm = bestk_style(ks)
        grid = best_k_grid(df, params, err, ks)
        draw_grid(ax_a, grid, cmap=cmap, norm=norm)
        annotate_grid(ax_a, grid)

    # --- examples / domain dots
    # if examples is not None:
    #     mark_examples(ax_a, examples, model, example_cmap)
    if fits is not None:
        ax_a.scatter(fits[params[0]], fits[params[1]], c=fits["share"], cmap="coolwarm",
                     vmin=0, vmax=1, s=100, linewidth=1, edgecolors="white", zorder=10)
        dy = 1 / (len(fits) + 2)
        ax_dom.text(0, 1, f"Dots: avg. fit parameters per domain\n('{condition}' condition)",
                    fontsize=6, va="top", transform=ax_dom.transAxes)
        ax_dom.text(0.0, 0.85-dy, f"Domains:", fontsize=6, va="top", transform=ax_dom.transAxes)
        for i, (nm, r) in enumerate(fits.iterrows()):
            ax_a.text(r[params[0]], r[params[1]], str(i), fontsize=6, ha="center", va="center", zorder=14)
            ax_dom.text(0.1, 0.85 - (i + 2.5) * dy, f"{i}: {nm}", fontsize=6, va="top", transform=ax_dom.transAxes)

    # --- single colorbar for a
    lab = []
    errTy = err.split('_')[-1]
    if len(ks) == 2:
        # lab.append(rf"background: sim. prob. $\theta_{r'{\rm ind}'}(k={k_lo})\leq\theta_{r'{\rm ind}'}(k={k_hi})$")
        lab.append(rf"Pixel Colour: Share of simulations with"+"\n"+rf"$\theta_{r'{\rm '}{errTy}{r'}'}(k={k_lo}) < \theta_{r'{\rm '}{errTy}{r'}'}(k={k_hi})$" + "\n")
    if fits is not None:
        lab.append(rf"Dot Colour: # of tasks with"+"\n"+rf"$\theta_{r'{\rm '}{errTy}{r'}'}^{r'{\rm Most\ recent}'} < \theta_{r'{\rm '}{errTy}{r'}'}^{r'{\rm Consensus}'}$")
    if lab:
        cb = fig.colorbar(ScalarMappable(mcolors.Normalize(0, 1), "coolwarm"), cax=cax_a)
        cb.set_label("\n".join(lab), fontsize=7)
        cb.set_ticks([0, 0.5, 1])
    else:
        cax_a.remove()

    # --- legend of examples
    # h, l = ax_a.get_legend_handles_labels()
    # if h:
    #     ax_leg.legend(h, l, loc="center left", frameon=False, fontsize=7)
    if fits is None:
        ax_dom.axis("off")
    ax_dom.axis("off")

    # --- axes cosmetics
    if model == "mh":
        ax_a.invert_xaxis()
        # ax_b.invert_xaxis()                       # c follows (shared)
    ax_a.set_xlabel(xlabel)
    ax_a.set_ylabel(ylabel)
    #ax_c.tick_params(labelleft=False)

    ax_a.set_title((rf"Mayer&Heck ($R={df.R.unique()[0]}$)" if model == "mh" else "Copy&Integrate")
                 + f" — {'individual' if 'ind' in errTy else errTy}{' median' if 'Median' in errTy else ''} error")
    return fig, dict(a=ax_a)#, b=ax_b, c=ax_c)


# ------------------------------------------------- simulation vs. data check
def sim_vs_data(df, fits, model, ks, N=100, err="relative_error_ind"):
    params = PHASE_PARAMS[model]
    prob = win_prob(df[df.N == N], params, err, ks[0], ks[-1])
    xs, ys = prob.columns.values, prob.index.values
    out = fits.copy()
    out["sim_win_prob"] = [prob.loc[ys[np.abs(ys - y).argmin()], xs[np.abs(xs - x).argmin()]]
                           for x, y in zip(fits[params[0]], fits[params[1]])]
    return out

#%%

# =============================================================== run
KS_PLOT = [0, 1, 3, 5, 7, 9, 15, 19, 29, 39, 49, 69, 99, 127]
n = 127

#%%

KS, N_EVAL, COND, other = [3,n], n, "Most recent", "Consensus"
tag = "3v127" if len(KS) == 2 else ""

errTy = "ind"
err = "relative_error_"+errTy
errSi = errTy+"_error"
# --- CI
fits_ci = domain_fits("processed_data/simoiu_fitsCI.csv", ["c", "alpha"], PHASE_PARAMS["ci"], COND, other=other, errSi=errSi)
fig, axs = phase_figure(errorsCI, KS, "ci", N=N_EVAL, examples=errorsCIex, example_cmap="Set1",
                        fits=fits_ci, condition=COND, err=err,
                        xlabel=r"$c$ = copy rate", ylabel=r"$\alpha$ = social signal influence")
fig.savefig(f"figs/phase_{errTy}Error_CI{tag}_SimoiuDomains{COND.replace(" ", "")}.png", dpi=600)

# --- MH
D_MODE = R_MODE = "fixed"
fits_mh = domain_fits(f"processed_data/simoiu_fitsMH_D{D_MODE}_R{R_MODE}.csv",
                      ["G", "E"], PHASE_PARAMS["mh"], COND, other=other, errSi=errSi)
fig, axs = phase_figure(errorsMH, KS, "mh", N=N_EVAL, examples=errorsMHex, example_cmap="tab10", fits=fits_mh, condition=COND,
                         err=err,
                        xlabel=r"$G$ = individuality = 1 - base copy rate",
                        ylabel=r"$E$ = lack of expertise")
fig.savefig(f"figs/phase_{errTy}Error_MH{tag}_SimoiuDomains{COND.replace(" ", "")}_Dfit{D_MODE}_Rfit{R_MODE}.png", dpi=600)

#%%

from Sim_202609_models_main import n_seeds,T, gfuncs, seeds, anchoring, R, sigG, sigE, SIG_CI, simulateMH, simMH, errors_from_estimatesMH, errors_from_estimates, simulate, simCI
from scipy.special import logit
from scipy.special import expit as logist

def win_prob(df, params, err, k_a, k_b):
    """P(err[k_a] <= err[k_b]) over seeds; returns grid (index=y, columns=x)."""
    if df.duplicated(params + ["seed", "k"]).any():
        warnings.warn("Several rows per (params, seed, k); using 'first'.")
    wide = df.pivot_table(index=params + ["seed"], columns="k", values=err, aggfunc="first")
    wide = wide[[k_a, k_b]].dropna()
    return (wide[k_a] < wide[k_b]).groupby(level=params).mean().rename("sim_win_prob").reset_index()#.unstack(params[0])

def sim_vs_data(fits, model, ks,  n, n_seeds, N_evaluate, err="relative_error_ind",T=0):
    params = PHASE_PARAMS[model]
    paramsToSim  = fits.reset_index()[["domain_name"]+params].values.tolist()
    if model=="ci":
        simparams = [(c,a) for d, c,a in paramsToSim]
        errors  = simCI(n,n_seeds,[N_evaluate],SIG_CI,T,ks,simparams)
    elif model=="mh":
        simparams = [(logit(E), logit(G)) for d,E,G in paramsToSim]
        errors = simMH(n,n_seeds,[N_evaluate],T,ks,simparams, D=3, verbose = False)
        errors["logGstarmean"] = logist(errors["Gstarmean"]).round(6)
        errors["logEstarmean"] = logist(errors["Estarmean"]).round(6)
        for v in ["G","E"]: fits[f"log{v}starmean"] = fits[f"log{v}starmean"].round(6)
    errors["relative_error_crowd"] = errors["relative_error_final"]
    errors.loc[errors.k==0, "relative_error_crowd"] = errors.loc[errors.k==0, "relative_error_wocMean"]
    wins = win_prob(errors, params, err, k_a = ks[0], k_b=ks[-1])
    out = fits.merge(wins, how="left", on=params)
    return out

def domain_fits(csv, raw_cols, phase_cols, condition="Most recent", other="Consensus", errSi="agg"):
    """Per-domain mean fitted params + share of tasks where `condition` wins."""
    f = pd.read_csv(csv)
    f = f[f.cond.isin([condition, other])]
    f["crowd_error"] = np.nan
    f.loc[f.cond=="Control", "crowd_error"] = f.loc[f.cond=="Control", "agg_error"]
    f.loc[f.cond!="Control", "crowd_error"] = f.loc[f.cond!="Control", "final_error"]
    best = f.pivot_table(index="task_id", columns="cond", values=errSi, aggfunc="first")[[condition, other]].idxmin(axis="columns").rename("bestConditionof2")
    f = f.join(best, on=["task_id"])
    tasks = f.groupby(["domain_name", "task_id", "cond"], as_index=False).first()
    share = (tasks["bestConditionof2"] == condition).groupby(tasks["domain_name"]).mean()
    out = f[f.cond == condition].groupby("domain_name")[raw_cols].mean()
    out.columns = phase_cols
    out["share"] = share
    return out

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16/2.54, 8/2.54))
for ax, (model, df, fits) in zip(axs, [("ci", errorsCI, fits_ci), ("mh", errorsMH, fits_mh)]):
    res = sim_vs_data(fits_ci if model=="ci" else fits_mh, model, KS, N_EVAL, n_seeds, N_EVAL, err=err, T=T)
    r = res[["share", "sim_win_prob"]].corr().iloc[0, 1]
    sns.regplot(res, x="share", y="sim_win_prob", ax=ax, scatter_kws={"s": 10})
    ax.text(0.5, 0.95, rf"$r={r:.2f}$", ha="center", va="top", transform=ax.transAxes)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="", ylabel="",
           title=f"Mayer&Heck (D {D_MODE}, R={R if R_MODE=='fixed' else 'NA'})" if model == "mh" else "Copy&Integrate")
    ax.set_aspect("equal")
axs[0].set_ylabel(f"simulated win probability \nof $k={KS[0]}$ vs. $k={KS[-1]}$", fontsize=9,)
fig.supxlabel(f"win share of chains in 'Most recent' ($k={KS[0]}$) vs. 'Consensus' ($k={KS[1]}$) condition\nin Simoiu et al. (2019) per domain", fontsize=9)
fig.suptitle(f"Model vs. Data: performance w.r.t. {'individual' if 'ind' in err else ('crowd' if 'crowd' in err else err)}{' median' if 'Median' in err else ''} error")
fig.tight_layout()
plt.savefig(f"figs/corrSimoiuModels_{errTy}.png", dpi=600)
# %%
