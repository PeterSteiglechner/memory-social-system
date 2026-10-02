#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st
import json
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize_scalar


plt.rcParams.update({"font.size":8})
#%%
answers = pd.read_csv("wisdom-of-crowds-master/data/original/answers.csv")
tasks = pd.read_csv("wisdom-of-crowds-master/data/original/tasks.csv")
domains = pd.read_csv("wisdom-of-crowds-master/data/original/domains.csv")
users = pd.read_csv("wisdom-of-crowds-master/data/original/users.csv")
#remove weird answers
#answers = answers.loc[answers.user_id != 2895]
#answers = answers.loc[(answers.user_id != 2875 ) | (answers.task_id!=1910)]
answers = answers.loc[answers.task_id!=10752]  # has no correct answer

answers = answers.merge(tasks[["task_id", "correct_answer"]], on="task_id", how="left" )
tasks = tasks.merge(domains[["domain_id", "domain_name", "answer_type", "knowledge_type"]], on="domain_id", how="left" )
answers = answers.merge(tasks[["task_id", "domain_name", "answer_type", "knowledge_type"]], on="task_id", how="left" )
users = users.rename(columns={"experimental_condition":"cond"})
answers = answers.merge(users[["user_id", "cond"]], on="user_id", how="left" )
answers = answers.dropna(subset=["answer"])
df = answers.loc[answers.answer_type=='open-ended']
df["correct_answer"] = df["correct_answer"].astype(float)
df["answer"] = df["answer"].astype(float)
df = df.sort_values(["task_id", "cond", "start_time"])  
def stdTrimmed(x, trim=0):
    q_low = np.percentile(x, q=trim/2, method="closest_observation")
    q_hi  = np.percentile(x,q=100-trim/2, method="closest_observation")
    return (x[(x < q_hi) & (x > q_low)]).std()
def meanTrimmed(x, trim=0):
    q_low = np.percentile(x, q=trim/2, method="closest_observation")
    q_hi  = np.percentile(x,q=100-trim/2, method="closest_observation")
    return (x[(x < q_hi) & (x > q_low)]).mean()
stds = df.loc[df.cond=="Control"].groupby("task_id")["answer"].agg(stdTrimmed, trim=0).rename("std_Control").reset_index()
cutPercentiles = [0.01,0.99]
confInt = df.loc[df.cond=="Control"].groupby("task_id")["answer"].describe(percentiles=cutPercentiles).reset_index().rename(columns={"1%":"ctrlLoPerc", "99%":"ctrlHiPerc"})
df = df.merge(stds, on="task_id", how="left")
df = df.merge(confInt, on="task_id", how="left")

def get_median(x, trim=0):
    if json.loads(x)=="Not enough data":
        return np.nan
    x = [float(x) for x in json.loads(x)]
    return np.median(x) if trim==0 else meanTrimmed(x, trim)  

df["xn"] = np.nan
df.loc[df.cond=="Most recent", "xn"] = df.loc[df.cond=="Most recent"].apply(lambda x: get_median(x['cues'], trim=0), axis=1)
df.loc[df.cond=="Consensus", "xn"] = df.loc[df.cond=="Consensus", "cues"].replace({"not_enough_data":np.nan}).astype(float)
df.loc[df.cond=="Most confident", "xn"] = df.loc[df.cond=="Most confident"].apply(lambda x: get_median(x['cues'], trim=0), axis=1)

crowdAnswers = df.groupby(["task_id", "cond"])["answer"].median().rename("crowdAnswer").reset_index()
df = df.merge(crowdAnswers.loc[crowdAnswers.cond=="Control"].rename(columns={"crowdAnswer":"controlCrowdAnswer"}).drop(columns="cond"), on=["task_id"], how="left")
df = df.merge(crowdAnswers, on=["task_id", "cond"], how="left")


# clean answers
# cuts = np.arange(5,200)
# plt.plot(cuts, [len(df.loc[(abs(df.answer-df.correct_answer)/df.std_Control)>cut]) for cut in cuts])

outliers = df.loc[(df.answer>=2*df['ctrlHiPerc'])]
#df.loc[(df.correct_answer<df['ctrlLoPerc'])].task_id.unique()
print(len(df.loc[(df.answer<=0.5*df['ctrlLoPerc'])]), len( df.loc[(df.answer>=2*df['ctrlHiPerc'])]))
dfCleaned = df.loc[(df.answer>0.5*df['ctrlLoPerc']) & (df.answer<2*df['ctrlHiPerc'])]

outliers[["task_id", "user_id", "cond", "cues", "correct_answer", "answer", "std_Control"]].to_csv("outliers.csv", float_format="%.2f")
print("fraction of outlirs (>25 control stds)",len(outliers)/len(df))
# %%

df["dev"] = (df["answer"] - df["correct_answer"]) / df["std_Control"]

def ind_error(x):
    return meanTrimmed(x.dropna().abs(), trim=0)
def agg_error(x):
    return np.abs(np.median(x.dropna()))
def final_error(x):
    return np.abs(x.dropna().iloc[-1])

ind_errors = df.groupby(["task_id", "cond"])["dev"].agg(ind_error).rename("ind_error")
agg_errors = df.groupby(["task_id", "cond"])["dev"].agg(agg_error).rename("agg_error")
final_errors = df.groupby(["task_id", "cond"])["dev"].agg(final_error).rename("final_error")

errors = pd.concat([ind_errors, agg_errors, final_errors], axis=1).reset_index().dropna()

err = "compareError"
errors["compareError"]  = np.nan
errors.loc[errors.cond=="Control", err] = errors.loc[errors.cond=="Control", "agg_error"] 
errors.loc[errors.cond!="Control", err] = errors.loc[errors.cond!="Control", "ind_error"] 
best = errors.pivot_table(index="task_id", columns="cond", values=err)[["Consensus", "Most recent"]].idxmin(axis="columns").rename("bestCondition")
errors = errors.join(best, on=["task_id"])
errors = errors.join(tasks.set_index("task_id")["domain_name"], on="task_id")
best.value_counts()

errors.groupby("cond").agg_error.mean()
#%%

def get_chain(task, cond):
    chain = df.loc[(df.task_id==task) & (df.cond==cond)][["start_time","answer", "xn", "correct_answer", "std_Control", "crowdAnswer"]].sort_values("start_time").reset_index(drop=True)
    return chain 

for tid in df.task_id.unique():
    truth = df.loc[df.task_id == tid, "correct_answer"].unique()[0]
    ctrl_dev = np.abs(get_chain(tid, "Control")["answer"] - truth).values
    for cond in ["Control", "Most recent", "Consensus", "Most confident"]:
        agg_dev = np.abs(get_chain(tid, cond)["crowdAnswer"].values[0]-truth)
        perfScore = np.mean(agg_dev<ctrl_dev)*100
        errors.loc[(errors.task_id==tid) & (errors.cond==cond), "percentileRank"] = perfScore
# %%
# Model: r = (m) with probability c else (alpha * m + (1-alpha) * p), where p is private belief, m is social signal in sequential estimation task, and r is the actual guess. alpha is a learning rate and c is copying probability
# we have data for r and m and we have 100 p's from a control condition (without social influence)


def get_private_givenAlpha(m, r, alpha):
    return (r - alpha * m) / (1.0 - alpha)
# r = alpha * m + (1-alpha) * p
# hence, p = (r - alpha * m) / (1-alpha) This is the private belief


def wass_for_alpha(alpha, m, r, control):
    return wasserstein_distance(control, get_private_givenAlpha(m, r, alpha))

def trim_mask(s, zmax=5):
    """Boolean mask keeping values within zmax trimmed-SDs of the trimmed mean."""
    sd = stdTrimmed(s, trim=0.)
    if np.isnan(sd) or sd == 0:
        return pd.Series(True, index=s.index)
    return ((s - meanTrimmed(s, trim=0)) / sd).abs() <= zmax

ALPHA_MAX = 0.98
def fit_alpha(m, r, control, grid_size=100, alpha_max=ALPHA_MAX):
    """
    m       : previous medians (treated data points)
    r       : answers given by those points
    control : answers from the control group (no anchor shown)
    """
    m, r = np.asarray(m, float), np.asarray(r, float)
    if len(m) < 10 or len(control) < 10:
        return np.nan, np.nan, None, None
    # 1) coarse grid search over [0, 1) (alpha = 1 is undefined)
    grid = np.linspace(0, alpha_max, grid_size)
    dists = np.array([wass_for_alpha(a, m, r, control) for a in grid])
    i = dists.argmin()
    best, best_d = grid[i], dists[i]

    # 2) refine locally
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid_size - 1)]
    res = minimize_scalar(wass_for_alpha, bounds=(lo, hi),
                          args=(m, r, control), method="bounded")
    if res.fun < best_d:
        best, best_d = res.x, res.fun
    return best, best_d, grid, dists

def estimate_c_alpha(xn, yn, control, grid_size=100):
    """xn, ans: arrays for ALL participants in a chain (copiers included)."""
    is_copy = (xn == yn)                      # use np.isclose if answers are floats from a slider
    c = is_copy.mean()
    nc = ~is_copy
    alpha, w, _, _ = fit_alpha(xn[nc], yn[nc], control, grid_size)
    return c, alpha, w


def safe_pct(x, q=(2.5, 97.5)):
    return np.percentile(x, q) if len(x) >= 20 else (np.nan, np.nan)


N_BOOT, MIN_N = 100, 20
rng = np.random.default_rng(0)

fits = []
indit_data = {}
for tid in df.task_id.unique():
    if tid in df.task_id.unique()[::10]: print(tid) 
    ctrl_raw = get_chain(task=tid, cond="Control")["answer"]
    control = ctrl_raw[trim_mask(ctrl_raw)].to_numpy(float)
    indit_data[tid] = control.tolist()

    for cond in ["Most recent", "Consensus"]:
        chain = get_chain(task=tid, cond=cond).dropna(subset=["xn"])
        chain = chain[trim_mask(chain["answer"])]
        xn  = chain["xn"].to_numpy(float)
        yn = chain["answer"].to_numpy(float)

        is_copy = (xn == yn)
        n_noncopy = int((~is_copy).sum())
        c = is_copy.mean()

        # c bootstrap: always possible
        c_b = [is_copy[rng.integers(0, len(xn), len(xn))].mean()
               for _ in range(N_BOOT)]
        c_lo, c_hi = np.percentile(c_b, [2.5, 97.5])

        # alpha: only if enough non-copiers
        if n_noncopy >= MIN_N:
            _, alpha, w_min = estimate_c_alpha(xn, yn, control)
            nc = ~is_copy
            w_null = wass_for_alpha(0.0, xn[nc], yn[nc], control)

            a_b = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(xn), len(xn))
                j = rng.integers(0, len(control), len(control))
                if (xn[i] != yn[i]).sum() < MIN_N:
                    continue
                a_b.append(estimate_c_alpha(xn[i], yn[i], control[j], grid_size=40)[1])
            a_lo, a_hi = safe_pct(a_b)
        else:
            alpha = w_min = w_null = a_lo = a_hi = np.nan

        fits.append(dict(
            task_id=tid, cond=cond,
            n=len(chain), n_noncopy=n_noncopy, n_control=len(control),
            c=c, c_lo=c_lo, c_hi=c_hi,
            alpha=alpha, alpha_lo=a_lo, alpha_hi=a_hi,
            at_boundary=(alpha < 0.005) or (alpha > ALPHA_MAX - 0.005),
            wasserstein_min=w_min, wasserstein_null=w_null,
        ))

fits_df = pd.DataFrame(fits)


#%%

# fits = []
# indit_data = {}
# for tid in df.task_id.unique():
#     if tid in df.task_id.unique()[::10]: print(tid) 
#     chain_control = get_chain(task=tid, cond="Control")["answer"]
#     lo, hi = chain_control.quantile([0.025, 0.975])
#     if not np.isnan(stdTrimmed(chain_control, trim=0)):
#         z = (chain_control - meanTrimmed(chain_control)) / stdTrimmed(chain_control, trim=0)
#         chain_control_trim = chain_control[z.abs() <= 5]
#     else:
#         chain_control_trim= chain_control
#     # chain_control_trim = chain_control[chain_control.between(lo, hi)]
#     indit_data[tid]=chain_control_trim.tolist()
#     for cond in ["Most recent", "Consensus"]:
#         chain = get_chain(task=tid, cond=cond)
#         chain = chain.dropna(subset=["xn"])
#         if not np.isnan(stdTrimmed(chain["answer"], trim=0)) and stdTrimmed(chain["answer"], trim=0)>0:
#             z = (chain["answer"] - meanTrimmed(chain["answer"], trim=0)) / stdTrimmed(chain["answer"], trim=0)
#             chain_trim = chain.loc[z.abs() <= 5]
#         else:
#             chain_trim= chain
#         copiers = chain_trim.loc[(chain_trim.xn==chain_trim.answer)]
#         noncopiers =  chain_trim.loc[~(chain_trim.xn==chain_trim.answer)]
#         c= len(copiers)/len(chain_trim)
#         alpha_hat, w_min, grid, dists = fit_alpha(noncopiers["xn"], noncopiers["answer"], chain_control_trim)
#         fits.append([tid, cond, c, alpha_hat, w_min])
# fits_df = pd.DataFrame(fits, columns=["task_id", "cond","c", "alpha", "wasserstein_min"])
# %%

# ---------------------
# -----   PLOTTING
# ---------------------

# err = "ind_error"
# errors2 = errors.merge(
#     fits_df, on=["task_id", "cond"], how="left"
# )
# errors2_byDomain = errors2.groupby(["domain_name", "cond"])[["c", err]].mean().reset_index()

# g = sns.lmplot(errors2_byDomain, x="c", y=err, hue="cond", markers="s", scatter_kws={"s":50, "alpha":0.5},)
# sns.scatterplot(errors2, x="c", y=err, hue="cond", size=1, alpha=0.5, ax=g.ax, legend=False)
# plt.ylabel(r"avg error $\theta_{\rm ind}$ [of 95%CI] per task")

#%%


# err = "percentileRank"
# errors2 = errors.merge(
#     fits_df, on=["task_id", "cond"], how="left"
# )
# errors2_byDomain = errors2.groupby(["domain_name", "cond"])[["c", err]].mean().reset_index()

# g = sns.lmplot(errors2_byDomain, x="c", y=err, hue="cond", markers="s", scatter_kws={"s":50, "alpha":0.5},)
# sns.scatterplot(errors2, x="c", y=err, hue="cond", size=1, alpha=0.5, ax=g.ax, legend=False)
# plt.ylabel(r"percentile rank per task")#$\theta_{\rm ind}$ [of 95%CI] per task")

#%%

# for col in ["percentileRank", "agg_error", "ind_error", "final_error"]:
#     ctrl = (
#         errors.loc[errors.cond=="Control", [col, "task_id"]]
#         .groupby(["task_id"]).first()   # "first" skips NaN, so it picks the Control value
#     ).reset_index()
#     errors[f"{col}_rel"] = errors[col]  - errors["task_id"].map(ctrl.set_index("task_id")[col])

#%%
# errcols =  [e+"_rel" for e in ["percentileRank", "agg_error", "ind_error", "final_error"]]
# a = errors.loc[~(errors.cond=="Control"), errcols+["task_id", "cond"]].melt(id_vars=["task_id", "cond"], value_name="performance", var_name="metric" )

# ax = plt.axes()
# ax = sns.stripplot(
#     data=a, x="cond", y="performance", hue="metric",
#     marker=".", dodge=True, alpha=0.4, legend=False
# )
# ax = sns.boxplot(
#     data=a, x="cond", y="performance", hue="metric",
#    legend=False, fliersize=0, fill=False, whis=0
# )
# sns.pointplot(
#     data=a, x="cond", y="performance", hue="metric",
#     marker="s", linestyle="none", markeredgecolor="k", markeredgewidth=1,
#     estimator="mean", errorbar=("ci", 95),
#     dodge=0.8 - 0.8 / len(errcols),  # matches stripplot's dodge
#     ax=ax, zorder=10, err_kws={"color":"grey","linewidth":3}
# )
# ax.axhline(0, color='k')
# # ax.set_ylim(-8,8)

# plt.figure()
# a = errors.loc[~(errors.cond=="Control"), errcols+["cond", "domain_name"]].groupby(["domain_name", "cond"])[errcols].mean().reset_index().melt(id_vars=["domain_name", "cond"], value_name="performance", var_name="metric" )
# ax = sns.stripplot(
#     data=a, x="cond", y="performance", hue="metric",
#     marker="o", dodge=True, alpha=0.4, legend=False
# )
# ax = sns.boxplot(
#     data=a, x="cond", y="performance", hue="metric",
#    legend=False, fliersize=0, fill=False, whis=0
# )
# sns.pointplot(
#     data=a, x="cond", y="performance", hue="metric",
#     marker="s", linestyle="none", markeredgecolor="k", markeredgewidth=1,
#     estimator="mean", errorbar=("ci", 95),
#     dodge=0.8 - 0.8 / len(errcols),  # matches stripplot's dodge
#     ax=ax, zorder=10, err_kws={"color":"grey","linewidth":3}
# )
# ax.axhline(0, color='k')
# ax.set_ylim(-10,10)

#%%



# %%
df2 = df.merge(fits_df, how="left", on=["task_id", "cond"])
df3 = df2.merge(errors, how="left", on=["task_id", "cond", "domain_name"])
# %%
# df3.loc[df3.domain_name=="calories"].pivot_table(index="task_id", columns="cond", values="c", aggfunc="first").describe()
print(df3.loc[df3.cond.isin(["Most recent", "Consensus"])].groupby(["domain_name", "cond"])[["c", "c_lo", "c_hi"] + ["alpha", "alpha_lo", "alpha_hi"]].agg("mean").round(2).reset_index().sort_values(["cond", "domain_name"]).to_markdown())
# %%
df3.to_csv("processed_data/simoiu_fitsCI.csv")
# %%
