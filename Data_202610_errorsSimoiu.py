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
tid = 10136
print(tasks.loc[tasks.task_id==tid])
df.loc[(df.task_id==tid) & (df.cond=="Most recent"), "answer"].describe()
#%%
def get_chain(task, cond):
    chain = df.loc[(df.task_id==task) & (df.cond==cond)][["start_time","answer", "xn", "correct_answer", "std_Control", "medianCrowdAnswer"]].sort_values("start_time").reset_index(drop=True)
    return chain 

# %%

df["diff_rel"] = (df["answer"] - df["correct_answer"])/ df["std_Control"]
df["dev_rel"] = (df["answer"] - df["correct_answer"]).abs() / df["std_Control"]

# %%

indErr = df.groupby(["domain_name", "task_id", "cond"])["dev_rel"].mean().rename("indErr").reset_index()
finErr = df.groupby(["domain_name", "task_id", "cond"])["dev_rel"].last().rename("finErr").reset_index()
wocErr = df.groupby(["domain_name", "task_id", "cond"])["diff_rel"].median().abs().rename("wocErr").reset_index()

errs = indErr.merge(finErr, how="left", on=["domain_name", "task_id", "cond"])
errs = errs.merge(wocErr, how="left", on=["domain_name", "task_id", "cond"])
errs["outErr"] = errs["finErr"]
errs.loc[errs.cond=="Control", "outErr"] = errs.loc[errs.cond=="Control", 'wocErr']

# %%

# sns.histplot(errs.loc[errs.cond=="Consensus", ["finErr", "wocErr", "indErr"]].diff(axis=1),binrange=(-2,2), bins=101)
# plt.gca().axvline(0)
# %%
print(errs.loc[errs.cond.isin(["Consensus", "Most recent"])].pivot_table(index="domain_name", columns="cond", values=["outErr", "indErr"], aggfunc="mean").to_latex(float_format="%.2f").replace("_", " "))
#%%
print(errs.loc[errs.cond.isin(["Consensus", "Most recent"])].pivot_table(index="domain_name", columns="cond", values=["outErr", "indErr"], aggfunc="mean").to_latex(float_format="%.2f").replace("_", " "))

# %%


metrics = ["outErr", "indErr"]

pt = (errs.loc[errs.cond.isin(["Consensus", "Most recent"])]
          .pivot_table(index="domain_name", columns="cond",
                       values=metrics, aggfunc="mean"))

# Most recent minus Consensus; negative = Most recent has smaller error
diff = pd.DataFrame({r"$\Delta "+m+r"$": pt[m]["Most recent"] - pt[m]["Consensus"] for m in metrics})

# Keep only cells where Most recent is better, drop domains with no wins
# wins = diff.where(diff < 0).dropna(how="all")

print(diff.to_latex(float_format="%.2f", na_rep="--").replace("_", " "))


# %%
better = (diff < 0).loc[(diff < 0).any(axis=1)]
tab = better.replace({True: r"\checkmark", False: ""})
tab.index = tab.index.str.replace("_", " ")
print(tab.to_latex(escape=False))
# %%

metrics = ["finErr", "indErr", "wocErr"]


sub = errs.loc[errs.cond.isin(["Consensus", "Most recent"])]

# one row per (domain, task), columns = (metric, cond)
pt = sub.pivot_table(index=["domain_name", "task_id"], columns="cond", values=metrics, aggfunc="first")

# sanity check: exactly one chain per cond per task
assert sub.groupby( ["domain_name", "task_id", "cond"]).size().eq(1).all()

# Most recent minus Consensus; negative = Most recent better
diff = pd.DataFrame({m: pt[m]["Most recent"] - pt[m]["Consensus"] for m in metrics})
win = diff < 0   # boolean, per (domain, task) x metric

summary = win.groupby("domain_name").sum()
summary["n_tasks"] = win.groupby("domain_name").size()
print(summary.to_latex().replace("_", " "))


# %%
