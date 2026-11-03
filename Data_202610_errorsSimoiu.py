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
df["start_time"] = pd.to_datetime(df["start_time"])
df = df.sort_values(["domain_name", "task_id", "cond", "start_time"])
def stdTrimmed(x, trim=0):
    q_low = np.percentile(x, q=trim/2, method="closest_observation")
    q_hi  = np.percentile(x,q=100-trim/2, method="closest_observation")
    return (x[(x < q_hi) & (x > q_low)]).std()
# def meanTrimmed(x, trim=0):
#     q_low = np.percentile(x, q=trim/2, method="closest_observation")
#     q_hi  = np.percentile(x,q=100-trim/2, method="closest_observation")
#     return (x[(x < q_hi) & (x > q_low)]).mean()
stds = df.loc[df.cond=="Control"].groupby("task_id")["answer"].agg(stdTrimmed, trim=0).rename("std_Control").reset_index()
df = df.merge(stds, on="task_id", how="left")
# cutPercentiles = [0.01,0.99]
# confInt = df.loc[df.cond=="Control"].groupby("task_id")["answer"].describe(percentiles=cutPercentiles).reset_index().rename(columns={"1%":"ctrlLoPerc", "99%":"ctrlHiPerc"})
# df = df.merge(confInt, on="task_id", how="left")

def get_median(x, trim=0):
    if json.loads(x)=="Not enough data":
        return np.nan
    x = [float(x) for x in json.loads(x)]
    return np.median(x)# if trim==0 else meanTrimmed(x, trim)  

df["xn"] = np.nan
df.loc[df.cond=="Most recent", "xn"] = df.loc[df.cond=="Most recent"].apply(lambda x: get_median(x['cues'], trim=0), axis=1)
df.loc[df.cond=="Consensus", "xn"] = df.loc[df.cond=="Consensus", "cues"].replace({"not_enough_data":np.nan}).astype(float)
df.loc[df.cond=="Most confident", "xn"] = df.loc[df.cond=="Most confident"].apply(lambda x: get_median(x['cues'], trim=0), axis=1)

medianCrowdAnswers = df.groupby(["task_id", "cond"])["answer"].median().rename("medianCrowdAnswer").reset_index()
df = df.merge(medianCrowdAnswers.loc[medianCrowdAnswers.cond=="Control"].rename(columns={"medianCrowdAnswer":"controlCrowdAnswer"}).drop(columns="cond"), on=["task_id"], how="left")
df = df.merge(medianCrowdAnswers, on=["task_id", "cond"], how="left")

#%%

def get_chain(task, cond):
    chain = df.loc[(df.task_id==task) & (df.cond==cond)][["start_time","answer", "xn", "correct_answer", "std_Control", "medianCrowdAnswer"]].sort_values("start_time").reset_index(drop=True)
    return chain 

#%%
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

n_peepsPerChain = df.groupby("task_id")["cond"].value_counts().reset_index()
# selected_tasks = [10140+l for l in range(20)][-7:-2] # age questions
selected_tasks = [l for l in range(10740, 10760) if not l == 10752]
fig, axs = plt.subplots(1,4, sharex=True, sharey=True, figsize=(18/2.54, 8/2.54))
thetas = []
conds = ['Control', 'Consensus', 'Most recent','Most confident'] #answers.experimental_condition.unique()
for ax, cond in zip(axs.flatten(), conds):
    n_peepsPerChainCond = n_peepsPerChain.loc[n_peepsPerChain["cond"]==cond]
    cols = plt.get_cmap("tab20b")
    thetas_med = []
    thetas_mean = []
    for ni, task in enumerate(selected_tasks):
        c = cols(ni)
        # taskname = tasks.loc[tasks.task_id==task]["prompt"].str.replace("How old is ", "").str.replace("?", "").iloc[0]
        taskname = tasks.loc[tasks.task_id==task]["path_to_media"].str.replace("prediction_movie_rating/", "").str.replace(".mp4", "").str.replace("_", " ").iloc[0]
        # taskname += fr" $\xi={xi:.1f}$" 
        chain = get_chain(task, cond)
        (chain["answer"] - chain["correct_answer"]).plot(ax=ax, label="_", color=c, alpha=0.1, lw=0, marker="o", markersize=1)
        ax.plot([np.median((chain["answer"] - chain["correct_answer"])[:k]) for k in range(len(chain["answer"]))], color=c, alpha=0.5)
        # chain["rollMean"].plot(ax=ax, color=c, label=taskname, alpha=0.5)
        # chain["rollMedian"].plot(ax=ax, color=c, ls="--", label="_", alpha=0.5)
        # --- right: histogram of language counts ---
        y_grid = np.linspace(min((chain["answer"] - chain["correct_answer"]).dropna())*1.2, max((chain["answer"] - chain["correct_answer"]).dropna())*1.2, 100)
        kde = gaussian_kde((chain["answer"] - chain["correct_answer"]).dropna())
        density = kde(y_grid)
        divider = make_axes_locatable(ax)
        hax = divider.append_axes("right", size="20%", pad=0.02, sharey=ax)
        # hax.hist(chain["relanswer"], bins=y_grid, orientation="horizontal", color=c, alpha=0.1)
        hax.plot(density, y_grid, color=c, alpha=0.8, lw=1)
        hax.axis("off")
    ax.set_title(cond)
    # ax.text(5,-94, fr"${r'\overline{\xi_{\rm mean}}'}={np.mean(xis_mean):.2f}$; ${r'\overline{\xi_{\rm median}}'}={np.mean(xis_med):.2f}$")
    ax.axhline(0, color="k", alpha=0.3)
    ax.set_ylim(-100,100)
    if ax in axs[:]:
        ax.set_xlabel(f"rater $n$")
    else:
        ax.set_xticks([])
axs[0].legend(fontsize=4, ncol=3, frameon=False, handlelength=1,columnspacing=1, labelspacing=0, borderpad=0)
axs[0].set_ylabel("Deviation from true rating [T]")
# fig.suptitle("'how-old-is-X' questions")
fig.suptitle("'Estimate movie rating on Rotten Tomatoes'")
fig.tight_layout()
plt.savefig("figs/rotten_tomatoes_ratings.png", dpi=600)


#%%
tid = 10136
print(tasks.loc[tasks.task_id==tid])
df.loc[(df.task_id==tid) & (df.cond=="Most recent"), "answer"].describe()
#%%
# %%

df["diff_rel"] = (df["answer"] - df["correct_answer"])/ df["std_Control"]
df["dev_rel"] = (df["answer"] - df["correct_answer"]).abs() / df["std_Control"]

# %%

indErr = df.groupby(["domain_name", "task_id", "cond"])["dev_rel"].mean().rename("indErr").reset_index()
finErr = df.groupby(["domain_name", "task_id", "cond"])["dev_rel"].last().rename("finErr").reset_index()
indMedianErr = df.groupby(["domain_name", "task_id", "cond"])["dev_rel"].median().rename("indMedianErr").reset_index()
wocErr = df.groupby(["domain_name", "task_id", "cond"])["diff_rel"].median().abs().rename("wocErr").reset_index()

errs = indErr.merge(finErr, how="left", on=["domain_name", "task_id", "cond"])
errs = errs.merge(indMedianErr, how="left", on=["domain_name", "task_id", "cond"])
errs = errs.merge(wocErr, how="left", on=["domain_name", "task_id", "cond"])
errs["crowdErr"] = errs["finErr"]
errs.loc[errs.cond=="Control", "crowdErr"] = errs.loc[errs.cond=="Control", 'wocErr']

# %%

# sns.histplot(errs.loc[errs.cond=="Consensus", ["finErr", "wocErr", "indErr"]].diff(axis=1),binrange=(-2,2), bins=101)
# plt.gca().axvline(0)
# %%
print(errs.loc[errs.cond.isin(["Consensus", "Most recent"])].pivot_table(index="domain_name", columns="cond", values=["crowdErr", "indMedianErr"], aggfunc="mean").to_latex(float_format="%.2f").replace("_", " "))
#%%
#%%
err = "crowdErr"
conds = ["Most recent", "Consensus"]
sub = errs.loc[errs.cond.isin(conds)].pivot_table(index="task_id", columns="cond", values=err, aggfunc="first")[conds]
wincol = f"{conds[0]} wins over {conds[1]}"
wins = (sub[conds[1]] - sub[conds[0]]>0).rename(wincol)
wins = pd.DataFrame(wins).reset_index().merge(sub.reset_index(), on="task_id")
fig = plt.figure(figsize=(16/2.54, 7/2.54))
sns.boxplot(wins.melt(id_vars=[wincol, "task_id"], var_name="cond", value_name=err), hue="cond", y=wincol, x=err, dodge=True, orient="h", fliersize=0, fill=False)
sns.stripplot(wins.melt(id_vars=[wincol, "task_id"], var_name="cond", value_name=err), hue="cond", y=wincol, x=err, dodge=True, orient="h", size=2, alpha=0.5, legend=False)
plt.gca().text(1.0,1.0, f"# outliers = {sum(wins[conds[0]]>10)} ({conds[0]}) and {sum(wins[conds[1]]>10)} ({conds[1]})", transform=plt.gca().transAxes, va="bottom", ha="right")
plt.xlim(0,5 if "crowd" in err else 15)
plt.xlabel("individual median error" if err == "indMedianErr" else ("crowd error" if "crowd" in err else err))
fig.tight_layout()
plt.savefig(f"figs/Simoiu_task{err}_wins_{conds[0].replace(" ","")}-{conds[1]}.png", dpi=300)
# %%


metrics = ["crowdErr", "indMedianErr"]

pt = (errs.loc[errs.cond.isin(["Consensus", "Most recent"])]
          .pivot_table(index="domain_name", columns="cond",
                       values=metrics, aggfunc="mean"))

# Most recent minus Consensus; negative = Most recent has smaller error
diff = pd.DataFrame({r"$\Delta "+m+r"$": pt[m]["Most recent"] - pt[m]["Consensus"] for m in metrics})

# Keep only cells where Most recent is better, drop domains with no wins
# wins = diff.where(diff < 0).dropna(how="all")

print(diff.to_latex(float_format="%.2f", na_rep="--").replace("_", " "))

#%%

err = "crowdErr"
conds = ["Consensus", "Control", "Most recent"]
sub = errs.loc[errs.cond.isin(conds)].pivot_table(index="task_id", columns="cond", values=err, aggfunc="first")[conds]
sub.mean(axis=0)

# %%
metrics = ["indMedianErr", "crowdErr"]


sub = errs.loc[errs.cond.isin(["Consensus", "Most recent"])]

# one row per (domain, task), columns = (metric, cond)
pt = sub.pivot_table(index=["domain_name", "task_id"], columns="cond", values=metrics, aggfunc="first")

# sanity check: exactly one chain per cond per task
assert sub.groupby( ["domain_name", "task_id", "cond"]).size().eq(1).all()

# Most recent minus Consensus; negative = Most recent better
diff = pd.DataFrame({m: pt[m]["Most recent"] - pt[m]["Consensus"] for m in metrics})
win = diff < 0   # boolean, per (domain, task) x metric

summary = win.groupby("domain_name").sum()
summary["nr_tasks"] = win.groupby("domain_name").size()
print(summary.to_latex().replace("_", " "))


# %%



