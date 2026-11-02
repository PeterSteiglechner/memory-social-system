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
    return (x[(x <= q_hi) & (x >= q_low)]).std()
def meanTrimmed(x, trim=0):
    q_low = np.percentile(x, q=trim/2, method="closest_observation")
    q_hi  = np.percentile(x,q=100-trim/2, method="closest_observation")
    return (x[(x <= q_hi) & (x >= q_low)]).mean()
stds = df.loc[df.cond=="Control"].groupby("task_id")["answer"].agg(stdTrimmed, trim=0).rename("std_Control").reset_index()
# cutPercentiles = [0.025,0.975]
# confInt = df.loc[df.cond=="Control"].groupby("task_id")["answer"].describe(percentiles=cutPercentiles).reset_index().rename(columns={"2.5%":"ctrlLoPerc", "97.5%":"ctrlHiPerc"})
df = df.merge(stds, on="task_id", how="left")
# df = df.merge(confInt, on="task_id", how="left")

def get_median(x, trim=0):
    if json.loads(x)=="Not enough data":
        return np.nan
    x = [float(x) for x in json.loads(x)]
    return np.median(x) if trim==0 else meanTrimmed(x, trim)  

df["xn"] = np.nan
df.loc[df.cond=="Most recent", "xn"] = df.loc[df.cond=="Most recent"].apply(lambda x: get_median(x['cues'], trim=0), axis=1)
df.loc[df.cond=="Consensus", "xn"] = df.loc[df.cond=="Consensus", "cues"].replace({"not_enough_data":np.nan}).astype(float)
df.loc[df.cond=="Most confident", "xn"] = df.loc[df.cond=="Most confident"].apply(lambda x: get_median(x['cues'], trim=0), axis=1)

medianCrowdAnswers = df.groupby(["task_id", "cond"])["answer"].median().rename("medianCrowdAnswer").reset_index()
df = df.merge(medianCrowdAnswers.loc[medianCrowdAnswers.cond=="Control"].rename(columns={"medianCrowdAnswer":"controlCrowdAnswer"}).drop(columns="cond"), on=["task_id"], how="left")
df = df.merge(medianCrowdAnswers, on=["task_id", "cond"], how="left")


# clean answers
fact = 10
n_before = len(df)

ctrl = df.loc[df.cond == "Control"].groupby("task_id")["answer"]
ref = ctrl.quantile([0.025, 0.975]).unstack()
ref.columns = ["ctrlLoPerc", "ctrlHiPerc"]
ref["nCtrl"] = ctrl.size()
ref["ctrlMed"] = ctrl.median()
ref["ctrlIQR"] = ctrl.quantile(0.75) - ctrl.quantile(0.25)

df = df.merge(ref.reset_index(), on="task_id", how="left")

c = df.correct_answer
# how far control answers reach from the truth (max, not min)
reach = pd.concat([(df.ctrlHiPerc - c).abs(),
                   (df.ctrlLoPerc - c).abs()], axis=1).max(axis=1)

# floor so the scale can never be 0: a fraction of the truth and of the control IQR
floor = pd.concat([0.05 * c.abs(), df.ctrlIQR], axis=1).max(axis=1)
scale = pd.concat([reach, floor], axis=1).max(axis=1)

# fallback for tasks with too few controls: use the pooled spread for that task
few = df.nCtrl.fillna(0) < 10
poolSd = df.groupby("task_id")["answer"].transform("std")
scale = scale.where(~few, pd.concat([scale, poolSd], axis=1).max(axis=1))

df["dist"] = (df.answer - c).abs()
df["isOutlier"] = df.dist >= fact * scale

print(df.assign(out=df.isOutlier)  # use the frame before filtering
      .groupby("task_id").out.mean().sort_values(ascending=False).head(10))

outliers = df[df.isOutlier]
df = df[~df.isOutlier].copy()
print(f"fraction of outliers (> {fact}x control reach): {len(outliers)/n_before:.3f}")

#%%
def get_chain(task, cond):
    chain = df.loc[(df.task_id==task) & (df.cond==cond)][["start_time","answer", "xn", "correct_answer", "std_Control", "medianCrowdAnswer"]].sort_values("start_time").reset_index(drop=True)
    return chain 

# %%

df["dev"] = (df["answer"] - df["correct_answer"]) / df["std_Control"]

def ind_error(x):
    return meanTrimmed(x.dropna().abs(), trim=0)  # trim=0 just returns the mean
def agg_error(x, func="median"):
    if func=="mean":
        return np.abs(meanTrimmed(x.dropna(), trim=0))
    else: 
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
# bestCtrl = errors.pivot_table(index="task_id", columns="cond", values=err)[["Consensus", "Most recent", "Control"]].idxmin(axis="columns").rename("bestConditionCtrl")
# errorsCtrl = errors.join(bestCtrl, on=["task_id"])
best = errors.pivot_table(index="task_id", columns="cond", values=err)[["Consensus", "Most recent"]].idxmin(axis="columns").rename("bestCondition")
errors = errors.join(best, on=["task_id"])
errors = errors.join(tasks.set_index("task_id")["domain_name"], on="task_id")
best.value_counts()


metrics = ["final_error", "ind_error", "agg_error"]
errs = errors.groupby(["domain_name", "task_id", "cond"])[metrics].mean().reset_index()
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


errors.groupby("cond").agg_error.mean()
#%%


# get the percentile rank of the crowd answer (the )
for tid in df.task_id.unique():
    truth = df.loc[df.task_id == tid, "correct_answer"].unique()[0]
    ctrl_dev = np.abs(get_chain(tid, "Control")["answer"] - truth).values
    for cond in ["Control", "Most recent", "Consensus", "Most confident"]:
        crowdAnswers = get_chain(tid, cond)["medianCrowdAnswer"]
        assert len(crowdAnswers.unique())==1
        agg_dev = np.abs(crowdAnswers.values[0]-truth)
        perfScore = np.mean(agg_dev<ctrl_dev)*100
        errors.loc[(errors.task_id==tid) & (errors.cond==cond), "percentileRank"] = perfScore
# %%
# ---------------------
# -----   FIT CI model
# ---------------------

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

#%%
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

df2 = df.merge(fits_df, how="left", on=["task_id", "cond"])
df3 = df2.merge(errors, how="left", on=["task_id", "cond", "domain_name"])
# %%
# df3.loc[df3.domain_name=="calories"].pivot_table(index="task_id", columns="cond", values="c", aggfunc="first").describe()
print(df3.loc[df3.cond.isin(["Most recent", "Consensus"])].groupby(["domain_name", "cond"])[["c", "c_lo", "c_hi"] + ["alpha", "alpha_lo", "alpha_hi"]].agg("mean").round(2).reset_index().sort_values(["cond", "domain_name"]).to_markdown())
# %%
df3.to_csv("processed_data/simoiu_fitsCI.csv")
# %%


fitsKontalexi= (
    pd.read_table('kontalexiFits.md', sep="|", header=0, index_col=1, skipinitialspace=True)
    .dropna(axis=1, how='all')
    .iloc[1:]
    )
fitsKontalexi.index = fitsKontalexi.index.str.strip()
myfit= df3.loc[df3.cond=="Consensus"].groupby("domain_name")[["c"]].mean().round(2)
fitsKontalexi = pd.DataFrame(fitsKontalexi).join(myfit) #, how="left")
fitsKontalexi.corr()

#%%
# ---------------------
# -----   FIT MH model  (population-level G, E; D fixed)
# ---------------------
from scipy.special import erf
from scipy.optimize import minimize
from scipy.stats import norm

# Assumptions:
#  - G, E are shared across raters

D_MODE = "fixed"          # "free" | "control" | "fixed"
R_MODE = "fixed"          # "free" | "fixed"
D_FIXED = 3.0            # only used if D_MODE == "fixed"
R_FIXED = 0.4            # only used if R_MODE == "fixed"
USE_CTRL_LL = True       # only for "free": add control answers ~ N(T, D*E) to the likelihood
STANDARDIZE = True       # rescale each task: (v - T) / SD(control); then D is in control-SD units,
                         # so a fixed D=3 means the same thing across tasks.
                         # If False, D_FIXED is in raw answer units.

E_MIN, R_MIN, EPS = 1e-3, 1e-3, 1e-9

def get_truth(tid):
    return float(df.loc[df.task_id == tid, "correct_answer"].iloc[0])

def mh_p_adjust(signal, T, G, E, D):
    """Probability of producing an own estimate (1 - copy probability)."""
    z = (signal - T) / (D * E * np.sqrt(2))
    return G + (1.0 - G) * erf(z)** 2

def mh_sigma(D, E, R):
    return R * D * E

def unpack(theta, D_mode, R_mode, sd_ctrl):
    G, E = theta[0], theta[1]
    if R_mode == "free":
        R = theta[2]
    else:
        R = R_FIXED
    if D_mode == "free":
        D = np.exp(theta[(3 if R_mode=="free" else 2)])
    else:
        D = sd_ctrl / E if D_mode == "control" else D_FIXED
    return G, E, R, D



def mh_negll(theta, x, y, is_copy, T, control, D_mode, R_mode, sd_ctrl):
    G, E, R, D = unpack(theta, D_mode, R_mode, sd_ctrl)
    sigma = mh_sigma(D, E, R)
    pa = np.clip(mh_p_adjust(x, T, G, E, D), EPS, 1 - EPS)

    ll = np.log(1.0 - pa[is_copy]).sum()                       # copies: point mass at x
    nc = ~is_copy
    mu = T + E * (x[nc] - T)                                   # anchoring
    ll += (np.log(pa[nc]) + norm.logpdf(y[nc], mu, sigma)).sum()
    if D_mode == "free" and USE_CTRL_LL:
        ll += norm.logpdf(control, T, D * E).sum()             # no-signal raters: N(T, D*E)
    return -ll


def fit_mh(x, y, T, control, D_mode="fixed", R_mode="fixed", starts=None):
    """x: social signals, y: answers (ALL participants, copiers included)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    is_copy = np.isclose(x, y)                                         # np.isclose if slider floats
    nan = (np.nan,) * 5 + (None,)
    if len(x) < 10 or (~is_copy).sum() < 5:
        return nan
    sd_ctrl = np.std(control)
    if not np.isfinite(sd_ctrl) or sd_ctrl <= 0:
        return nan

    bounds = [(0.0, 1.0), (E_MIN, 1.0)]
    if R_mode=="free":
        bounds.append((R_MIN, 1))
    if D_mode == "free":
        bounds.append((np.log(0.05 * sd_ctrl), np.log(100 * sd_ctrl)))
    if starts is None:
        Rs = (0.3, 0.8) if R_mode == "free" else (None,)
        Ds = (1.0, 3.0) if D_mode == "free" else (None,)
        starts = []
        for G in (0.1, 0.5, 0.9):
            for E in (0.1, 0.5, 0.9):
                for R in Rs:
                    for D in Ds:
                        s = [G, E]
                        if R is not None: s.append(R)
                        if D is not None: s.append(np.log(D * sd_ctrl))
                        starts.append(tuple(s))
                    

    best = None
    for s in starts:
        r = minimize(mh_negll, s, args=(x, y, is_copy, T, control, D_mode, R_mode, sd_ctrl),
                     method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    G, E, R, D = unpack(best.x, D_mode, R_mode, sd_ctrl)
    return G, E, R, D, -best.fun, best.x                       # best.x = warm start for bootstrap

def ppc_wass(x, y, T, G, E, R, D, rng, n_rep=20):
    sigma = mh_sigma(D, E, R)
    out = []
    for _ in range(n_rep):
        adj = rng.random(len(x)) < mh_p_adjust(x, T, G, E, D)
        y_sim = np.where(adj, T + E * (x - T) + rng.normal(0, sigma, len(x)), x)
        out.append(wasserstein_distance(y, y_sim))
    return float(np.mean(out))

#%%
N_BOOT, MIN_N = 100, 20
rng = np.random.default_rng(0)

mh_fits = []
for tid in df.task_id.unique():
    if tid in df.task_id.unique()[::10]: print(f"run task {tid} ({(df.task_id.unique().tolist().index(tid)/len(df.task_id.unique()) * 100):.1f}%)")
    T_raw = get_truth(tid)
    ctrl_raw = get_chain(task=tid, cond="Control")["answer"]
    control_raw = ctrl_raw[trim_mask(ctrl_raw)].to_numpy(float)
    scale = np.std(control_raw) if STANDARDIZE else 1.0
    if not np.isfinite(scale) or scale <= 0: scale = 1.0
    T = 0.0 if STANDARDIZE else T_raw
    shift = T_raw if STANDARDIZE else 0.0
    control = (control_raw - shift) / scale

    for cond in ["Most recent", "Consensus"]:
        chain = get_chain(task=tid, cond=cond).dropna(subset=["xn"])
        chain = chain[trim_mask(chain["answer"])]
        xn = (chain["xn"].to_numpy(float) - shift) / scale
        yn = (chain["answer"].to_numpy(float) - shift) / scale

        is_copy = (xn == yn)
        n_noncopy = int((~is_copy).sum())
        c_emp = is_copy.mean()

        nanrow = dict.fromkeys(["G","E","R","D","ll","c_model","wass_ppc",
                                "G_lo","G_hi","E_lo","E_hi","R_lo","R_hi","D_lo","D_hi"], np.nan)
        row = nanrow
        if n_noncopy >= MIN_N:
            G, E, R, D, ll, th = fit_mh(xn, yn, T, control, D_mode=D_MODE, R_mode=R_MODE)
            if np.isfinite(G):
                row = dict(G=G, E=E, R=R, D=D, ll=ll,
                           c_model=float(np.mean(1 - mh_p_adjust(xn, T, G, E, D))),
                           wass_ppc=ppc_wass(xn, yn, T, G, E, R, D, rng))
                b = []
                for _ in range(N_BOOT):
                    i = rng.integers(0, len(xn), len(xn))
                    j = rng.integers(0, len(control), len(control))
                    if (xn[i] != yn[i]).sum() < MIN_N:
                        continue
                    out = fit_mh(xn[i], yn[i], T, control[j], D_mode=D_MODE, R_mode=R_MODE, starts=[tuple(th)])
                    if np.isfinite(out[0]):
                        b.append(out[:4])
                b = np.array(b) if len(b) else np.empty((0, 4))
                for k, name in enumerate(["G", "E", "R", "D"]):
                    lo, hi = safe_pct(b[:, k]) if len(b) else (np.nan, np.nan)
                    row[f"{name}_lo"], row[f"{name}_hi"] = lo, hi
                row = {**nanrow, **row}

        g, e, r_, d = row["G"], row["E"], row["R"], row["D"]
        mh_fits.append(dict(
            task_id=tid, cond=cond, T=T_raw, d_mode=D_MODE, r_mode=R_MODE, standardized=STANDARDIZE,
            n=len(chain), n_noncopy=n_noncopy, n_control=len(control), c_emp=c_emp,
            c_model=row["c_model"],
            G=g, G_lo=row["G_lo"], G_hi=row["G_hi"],
            E=e, E_lo=row["E_lo"], E_hi=row["E_hi"],
            R=r_, R_lo=row["R_lo"], R_hi=row["R_hi"],
            D=d, D_lo=row["D_lo"], D_hi=row["D_hi"],
            sigma=mh_sigma(d, e, r_),
            at_boundary=bool(np.isfinite(e) and (
                e < E_MIN + .005 or e > .995 or r_ < R_MIN + .005 or r_ > .995 or g < .005 or g > .995)),
            loglik=row["ll"], wasserstein_ppc=row["wass_ppc"],
        ))

mh_fits_df = pd.DataFrame(mh_fits)

#%%
mh_fits_df

#%%
df2MH = df.merge(mh_fits_df, how="left", on=["task_id", "cond"])
df3MH = df2MH.merge(errors, how="left", on=["task_id", "cond", "domain_name"])
# %%
cols = np.concatenate([[k, k+"_lo", k+"_hi"] for k in ["G", "E", "D", "R"]])
print(df3MH.loc[df3MH.cond.isin(["Most recent", "Consensus"])].groupby(["domain_name", "cond"])[cols].agg("mean").round(2).reset_index().sort_values(["cond", "domain_name"]).to_markdown())
# %%
df3MH.to_csv(f"processed_data/simoiu_fitsMH_D{D_MODE}_R{R_MODE}.csv")

#%%
domainMapper = {tid:(tasks.loc[tasks.task_id==tid, "domain_name"].iloc[0]) for tid in mh_fits_df.task_id.unique()}
mh_fits_df["domain"] = mh_fits_df["task_id"].replace(domainMapper)
mh_fits_df.groupby(["cond", "domain"])["E"].describe()

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
ci_fits = pd.read_csv(f"processed_data/simoiu_fitsCI.csv")

mh_fits = pd.read_csv(f"processed_data/simoiu_fitsMH_Dfixed_Rfixed.csv")
#%%

alpha_corr = ci_fits.groupby(["task_id", "cond"])["alpha"].first().dropna().reset_index().pivot_table(columns="cond", index="task_id", values="alpha").corr()
c_corr = ci_fits.groupby(["task_id", "cond"])["c"].first().dropna().reset_index().pivot_table(columns="cond", index="task_id", values="c").corr()
print(f"correaltions for c and alpha across tasks in consnesus/most-recent: corr-c = {c_corr.iloc[0,1]:.2f}; corr-alpha = {alpha_corr.iloc[0,1]:.2f} ")
# %%

G_corr = mh_fits.groupby(["task_id", "cond"])["G"].first().dropna().reset_index().pivot_table(columns="cond", index="task_id", values="G").corr()
E_corr = mh_fits.groupby(["task_id", "cond"])["E"].first().dropna().reset_index().pivot_table(columns="cond", index="task_id", values="E").corr()
print(f"correaltions for E and G across tasks in consnesus/most-recent: corr-E = {E_corr.iloc[0,1]:.2f}; corr-G = {G_corr.iloc[0,1]:.2f} ")

# %%
params = ["c", "alpha"]
print(ci_fits.groupby(["cond"])[params].agg(["mean", "std"]).dropna().to_latex(float_format="%.2f"))

# %%
params = ["G", "E"]
print(mh_fits.groupby(["cond"])[params].agg(["mean", "std"]).dropna().to_latex(float_format="%.2f"))
# %%
