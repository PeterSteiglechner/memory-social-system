#%% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st
import json
from scipy.stats import gaussian_kde
import statsmodels
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({"font.size":8})
deviationThresh = 0.2

answers = pd.read_csv("wisdom-of-crowds-master/data/original/answers.csv")
tasks = pd.read_csv("wisdom-of-crowds-master/data/original/tasks.csv")
domains = pd.read_csv("wisdom-of-crowds-master/data/original/domains.csv")
users = pd.read_csv("wisdom-of-crowds-master/data/original/users.csv")

answers = answers.merge(tasks[["task_id", "correct_answer"]], on="task_id", how="left" )
tasks = tasks.merge(domains[["domain_id", "domain_name", "answer_type", "knowledge_type"]], on="domain_id", how="left" )
answers = answers.merge(tasks[["task_id", "domain_name", "answer_type", "knowledge_type"]], on="task_id", how="left" )
answers = answers.merge(users[["user_id", "experimental_condition"]], on="user_id", how="left" )

variance_open_tasks = answers.loc[(answers.answer_type=="open-ended") & (answers.experimental_condition=="Control")].assign(answer_float = lambda x: x['answer'].astype(float)).groupby("task_id")["answer_float"].std()
tasks = tasks.merge(variance_open_tasks, on="task_id", how="left").rename(columns={"answer_float":"std_answer"})
tasks["rel_std_answer"] = np.nan
tasks.loc[(tasks.answer_type=="open-ended"), "rel_std_answer"] = tasks.loc[(tasks.answer_type=="open-ended"), "std_answer"]/tasks.loc[(tasks.answer_type=="open-ended"), "correct_answer"].astype(float)
answers = answers.merge(tasks[["task_id", "std_answer", "rel_std_answer"]], on="task_id", how="left")
answers.loc[answers.answer_type=="discrete", "correct"] = (answers.loc[answers.answer_type=="discrete", "correct_answer"]==answers.loc[answers.answer_type=="discrete", "answer"]).astype(bool)
answers.loc[answers.answer_type=="open-ended", "correct"] = (abs(answers.loc[answers.answer_type=="open-ended", "correct_answer"].astype(float) - answers.loc[answers.answer_type=="open-ended", "answer"].astype(float))<deviationThresh * answers.loc[answers.answer_type=="open-ended", "std_answer"]).astype(bool)


# %%
selection = "answer_type=='open-ended'"
n_peepsPerChain = answers.query(selection).groupby("task_id")["experimental_condition"].value_counts().reset_index()
#%%
n_peepsPerChainControl = n_peepsPerChain.loc[n_peepsPerChain["experimental_condition"]=="Control"]
fig, ax = plt.subplots(1,1)
cols = plt.get_cmap("tab10")
for ni, i in enumerate(range(len(n_peepsPerChainControl))[5:12]):
    c = cols(ni)
    task, cond = n_peepsPerChainControl.iloc[i][["task_id", "experimental_condition"]].tolist()
    chain = answers.loc[(answers.task_id==task) & (answers.experimental_condition==cond)][["start_time","answer", "cues", "correct_answer", "std_answer"]].sort_values("start_time")
    answer_dev = (chain["answer"].astype(float) - chain["correct_answer"].astype(float))/chain["correct_answer"].astype(float)
    answer_dev = answer_dev.reset_index(drop=True)
    answer_dev.plot(ax=ax, label=task, color=c, alpha=0.2)
    answer_dev.expanding().mean().plot(ax=ax, label="task", color=c)
    answer_dev.expanding().median().plot(ax=ax, label="task", color=c, ls="--")
ax.set_title("Control")
ax.axhline(0, color="k")
ax.set_ylim(-3,3)

# %%
def get_chain(task, cond):
    chain = answers.loc[(answers.task_id==task) & (answers.experimental_condition==cond)][["start_time","answer", "cues", "correct_answer", "std_answer"]].sort_values("start_time").reset_index(drop=True)
    chain["relanswer"] = (chain["answer"].astype(float) - float(chain["correct_answer"].iloc[0]))
    chain.loc[chain["relanswer"].abs()>100, "relanswer"] = np.nan
    rollMean = chain["relanswer"].expanding().mean().rename("rollMean")
    rollMedian = chain["relanswer"].expanding().median().rename("rollMedian")
    chain = chain.join(rollMean)
    chain = chain.join(rollMedian)
    return chain 


# selected_tasks = [10140+l for l in range(20)][-7:-2] # age questions
selected_tasks = [l for l in range(10740, 10760) if not l == 10752]
fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(18/2.54, 14/2.54))
xis = []
conds = ['Control', 'Consensus', 'Most recent','Most confident'] #answers.experimental_condition.unique()
for ax, cond in zip(axs.flatten(), conds):
    n_peepsPerChainCond = n_peepsPerChain.loc[n_peepsPerChain["experimental_condition"]==cond]
    cols = plt.get_cmap("tab20b")
    xis_med = []
    xis_mean = []
    for ni, task in enumerate(selected_tasks):
        c = cols(ni)
        # taskname = tasks.loc[tasks.task_id==task]["prompt"].str.replace("How old is ", "").str.replace("?", "").iloc[0]
        taskname = tasks.loc[tasks.task_id==task]["path_to_media"].str.replace("prediction_movie_rating/", "").str.replace(".mp4", "").str.replace("_", " ").iloc[0]

        xi_mean = (abs(chain.answer.astype(float).mean()- chain.correct_answer.unique().astype(float))/chain.std_answer.unique())[0]
        xis_mean.append(xi_mean)
        xi_median = (abs(chain.answer.astype(float).median()- chain.correct_answer.unique().astype(float))/chain.std_answer.unique())[0]
        xis_med.append(xi_median)
        xis.append([cond, taskname, xi_mean, xi_median])

        # taskname += fr" $\xi={xi:.1f}$" 
        chain = get_chain(task, cond)
        chain["relanswer"].plot(ax=ax, label="_", color=c, alpha=0.1, lw=0, marker="o", markersize=1)
        chain["rollMean"].plot(ax=ax, color=c, label=taskname, alpha=0.5)
        chain["rollMedian"].plot(ax=ax, color=c, ls="--", label="_", alpha=0.5)
        # --- right: histogram of language counts ---
        y_grid = np.linspace(min(chain["relanswer"].dropna())*1.2, max(chain["relanswer"].dropna())*1.2, 100)
        kde = gaussian_kde(chain["relanswer"].dropna())
        density = kde(y_grid)
        divider = make_axes_locatable(ax)
        hax = divider.append_axes("right", size="20%", pad=0.02, sharey=ax)
        # hax.hist(chain["relanswer"], bins=y_grid, orientation="horizontal", color=c, alpha=0.1)
        hax.plot(density, y_grid, color=c, alpha=0.8, lw=1)
        hax.axis("off")
    ax.set_title(cond)
    ax.text(5,-94, fr"${r'\overline{\xi_{\rm mean}}'}={np.mean(xis_mean):.2f}$; ${r'\overline{\xi_{\rm median}}'}={np.mean(xis_med):.2f}$")
    ax.axhline(0, color="k", alpha=0.3)
    ax.set_ylim(-100,100)
    if ax in axs[1,:]:
        ax.set_xlabel(f"rater $n$")
    else:
        ax.set_xticks([])
axs[0,0].legend(fontsize=4, ncol=3, frameon=False, handlelength=1,columnspacing=1, labelspacing=0, borderpad=0)
axs[0,0].set_ylabel("Deviation from true rating [T]", y=0)
# fig.suptitle("'how-old-is-X' questions")
fig.suptitle("'Estimate movie rating on Rotten Tomatoes'")
fig.tight_layout()
plt.savefig("figs/rotten_tomatoes_ratings.png", dpi=600)

xis_df = pd.DataFrame(xis, columns=["condition", "movie", "xi_mean", "xi_median"])
# %%
fig, ax = plt.subplots(1,1,figsize=(8/2.54, 8/2.54))
xis_df_melted = xis_df.melt(id_vars=["condition", "movie"], value_name="xi", var_name="metric").replace({"xi_mean":"mean", "xi_median":"median"})
sns.barplot(xis_df_melted, hue="condition", x="metric", y="xi", saturation=0.2)
sns.stripplot(xis_df_melted, hue="condition", x="metric", y="xi", dodge=True, legend=False, size=2)
plt.xlabel(None)
plt.legend(title="", frameon=False, ncol=1)
plt.ylabel(r"relative error $\xi$")
plt.title("Collective performance\n'Estimate movie rating on Rotten Tomatoes'")
fig.tight_layout()
# %%
selected_tasks = [10740+l for l in range(10)]
results = []
for task in selected_tasks:
    cond = "Control"
    chainControl = get_chain(task, cond)       
    # taskname = tasks.loc[tasks.task_id==task]["prompt"].str.replace("How old is ", "").str.replace("?", "").iloc[0]
    taskname = tasks.loc[tasks.task_id==task]["path_to_media"].str.replace("prediction_movie_rating/", "").str.replace(".mp4", "").str.replace("_", " ").iloc[0]
    mu_p = np.mean(chainControl["relanswer"])
    sigma_p = np.std(chainControl["relanswer"], ddof=1)

    cond = "Most recent"
    def get1from3cues(x):
        if type(x)==float:
            return x
        else:
            return np.mean([float(c) for c in json.loads(x)])

    chainCond = get_chain(task, cond)
    chainCond["cues"] = chainCond["cues"].astype(str).replace('"Not enough data"', np.nan).replace("not_enough_data", np.nan)
    if "Most" in cond:
        chainCond["cues"] = chainCond["cues"].apply(get1from3cues)
    chainCond["cues"] = chainCond["cues"].astype(float) - chainCond["correct_answer"].astype(float)

    chainCond = chainCond.dropna(subset=["cues", "answer", "correct_answer"])


    def neg_log_lik(params, x_social, y):
        theta, log_sigma_e = params
        alpha = 1 / (1 + np.exp(-theta))         
        sigma_e = np.exp(log_sigma_e)
        mean = alpha * x_social + (1 - alpha) * mu_p
        var = (1 - alpha)**2 * sigma_p**2 + sigma_e**2
        return -np.sum(st.norm.logpdf(y, loc=mean, scale=np.sqrt(var)))
    
    x_social = chainCond["cues"]
    y = chainCond["answer"].astype(float) - chainCond["correct_answer"].astype(float)
    res = minimize(neg_log_lik, x0=[0.0, 0.0], args=(x_social, y), method="BFGS")
    theta_hat, log_sigma_e_hat = res.x
    alpha_hat = 1 / (1 + np.exp(-theta_hat))
    y_hat = alpha_hat * x_social + (1 - alpha_hat) * mu_p
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    results.append([taskname, alpha_hat, r2, mu_p, sigma_p])


    # results.append([taskname, alpha_hat, r2])
print(pd.DataFrame(results, columns=["task", "alpha", "r2", "mu_p", "sigma_p"]).set_index("task").sort_values("alpha").to_string(float_format="%.2f", index=True, index_names=True, col_space=20, justify="right"))
# %%
