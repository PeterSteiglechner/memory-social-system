#%% 
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.special import expit as logist
from scipy.special import logit

SQ2 = 2 ** 0.5

# %%
# 
n_seeds = 1000
n = 127  # raters
N_evaluate = np.array([10,20,30,40,50,75,100,150,200,250,300,400,500,600,700,800,900,1000])
N_evaluate = [N for N in N_evaluate if N<=n]
N_evaluate = [n]
inds = np.arange(n)
T = 0.2
gfuncs = ["median"]
seeds = np.arange(n_seeds)

# %%
# -----   RUN MAYER AND HECK
# ---------------------

rng = np.random.default_rng(0)

R = 0.4
anchoring = True
sigG, sigE = (1e-5, 1e-5)

def simulateMH(k, Estar, Gstar, D, gfunc, gauss_noise, unif_noise):
    """All seeds at once. Estar, G, gauss_noise, u have shape (n_seeds, n)."""
    E = logist(Estar)
    estimates = np.empty((n_seeds, n))
    estimates[:, 0] = T + D * E[:, 0] * gauss_noise[:, 0]                      # first guess: no signal
    agg = np.mean if gfunc == "mean" else np.median
    G = logist(Gstar)                                            # precompute
    for i in range(1, n):
        signal = agg(estimates[:, max(0, i - k):i], axis=1)
        p_copy = 1 - (G[:, i] + (1 - G[:, i]) *
                      erf((signal - T) / (D * E[:, i] * SQ2)) ** 2)
        copy = unif_noise[:, i] < p_copy
        loc = T + anchoring * E[:, i] * (signal - T)
        new = loc + R * D * E[:, i] * gauss_noise[:, i]
        estimates[:, i] = np.where(copy, signal, new)
    return estimates

def errors_from(estimates, Estar, D):
    E = logist(Estar)
    wocMean_err, ind_err, exp_err, fin_err = [], [], [], []
    for N in N_evaluate:
        avgSigma = (D * E[:, :N]).mean(axis=1)
        est = estimates[:, :N]                              # guesses 0..N
        dev = np.abs(est - T)
        wocMean_err.append(np.abs(np.median(est, axis=1) - T) / avgSigma)
        ind_err.append(dev.mean(axis=1) / avgSigma)
        # exp_err.append(dev.mean(axis=1) / (D * logist(E[:, :N + 1].mean(axis=1))))
        fin_err.append(dev[:, -1] / avgSigma)
    # each has shape (n_seeds, len(N_evaluate))
    return tuple(np.stack(errors, axis=1) for errors in (wocMean_err, ind_err, fin_err))

#%%
run = ""
if run=="_examples":
    memory_sizes = [0] + list(range(1, min(n, 100), 2))+ [127,149,199,249,299] + list(np.arange(399,1000,100))
    memory_sizes = [k for k in memory_sizes if k<=n]
    #MHparams = [(-1,3,0), (-1,3,-2), (-1,3,2), (0,3,0), (-1,1,0)] # (-4,3,-4), (4,3,-4), (-4,3,4), (4,3,4), 
    MHparams = [(-1,-1), (1,-1), (-1,1), (1,1)]
else:
    memory_sizes = [0,1,3,9,19,99,127]
    Gstars = logit(np.linspace(logist(-4), logist(4), 21))
    Estars = logit(np.linspace(logist(-4), logist(4), 21))
    MHparams = [(G,E) for G in Gstars for E in Estars]

def simMH(n, n_seeds, N_evaluate, T, memory_sizes, paramsMH, D=3, verbose=True):
    rows = []
    for Gstarmean, Estarmean in paramsMH:
        Estar = rng.normal(Estarmean, sigE, (n_seeds, n))
        Gstar = rng.normal(Gstarmean, sigG, (n_seeds, n))
        for gfunc in gfuncs:
            if verbose and (paramsMH.index((Gstarmean, Estarmean)) % 5 == 0): print(".", end=" ")
            for k in memory_sizes:
                gauss_noise = rng.standard_normal((n_seeds, n))
                unif_noise = rng.random((n_seeds, n))
                if k == 0:
                    estimates = T + D * logist(Estar) * gauss_noise
                    #wocMean_err, ind_err, fin_err = errors_from(estimates, E, D)
                else:
                    estimates = simulateMH(k, Estar, Gstar, D, gfunc, gauss_noise, unif_noise)
                wocMean_err, ind_err, fin_err = errors_from(estimates, Estar, D)

                rows.append(pd.DataFrame({
                    "seed": np.repeat(seeds, len(N_evaluate)),
                    "N": np.tile(N_evaluate, n_seeds),
                    "n": n, "gfunc": gfunc, "D": D,
                    "Estarmean": Estarmean, "Gstarmean": Gstarmean, "R": R, "k": k,
                    "relative_error_wocMean": wocMean_err.ravel(),
                    "relative_error_ind": ind_err.ravel(),
                    # "relative_error_ind": e_exp.ravel(),
                    "relative_error_final": fin_err.ravel()}))
    errorsMH = pd.concat(rows, ignore_index=True)
    # errorsMH = errorsMH.rename(columns = {"relative_error_ind_expectedSigma":"relative_error_ind"})
    errorsMH.loc[errorsMH["k"]>errorsMH["N"], [c for c in errorsMH.columns if "relative_error" in c]] = np.nan
    return errorsMH

#%%

if __name__=="__main__":
    errorsMH = simMH(n, n_seeds, N_evaluate, T, memory_sizes, MHparams)
    print("\n" + f"simData/mh{run}{f'_N{n}' if n!=100 else ''}.csv")
    # (errorsMH.loc[errorsMH.N==n]).to_csv(f"simData/mh{run}{f'_N{n}' if n!=100 else ''}.csv", index=False)

#%%
#%%
# ---------------------
# -----   Copy and Integrate by Marina, Alexandros, Pantelis et al
# ---------------------

rng = np.random.default_rng(0)

SIG_CI = 1

run = ""
if run=="_examples":
    memory_sizes = [0] + list(range(1, min(n, 100), 2))+ [127,149,199,249,299] + list(np.arange(399,1000,100))
    memory_sizes = [k for k in memory_sizes if k<=n]
    cs = [0.2,0.8]
    alphas = [0.2,0.7]
    paramsCI = [(c,a) for c in cs for a in alphas]
else:
    memory_sizes = [0,1,3,9,19,99,127]
    cs = np.linspace(0,1,21)#[1:-1]
    alphas = np.linspace(0,1,21)#[1:-1]
    paramsCI = [(c,a) for c in cs for a in alphas]

def simulate(k, c, alpha, gfunc, beliefs, uniformRand):
    estimates = np.empty((n_seeds, n))
    estimates[:, 0] = beliefs[:, 0]
    copy = uniformRand < c            # all coin flips in one draw
    agg = np.mean if gfunc == "mean" else np.median
    for i in range(1, n):
        signal = agg(estimates[:, max(0, i - k):i], axis=1)
        estimates[:, i] = np.where(copy[:, i], signal,
                           alpha * signal + (1 - alpha) * beliefs[:, i])
    return estimates

def metrics(g):
    wocMean_err, ind_err, fin_err = [], [], []
    for N in N_evaluate:
        est = g[:, :N]   # guesses 0..N
        dev = np.abs(est - T)
        wocMean_err.append(np.abs(np.median(est, axis=1) - T) /SIG_CI)
        ind_err.append((dev / SIG_CI).mean(axis=1))
        fin_err.append(dev[:, -1] / (SIG_CI))
    return tuple(np.stack(a, axis=1) for a in (wocMean_err, ind_err, fin_err))

def simCI(n, n_seeds, N_evaluate, sig, T, memory_sizes, paramsCI):
    beliefs = rng.normal(T, sig, size=(n_seeds, n))   # generated once, shared by all configs
    u = rng.random((n_seeds, n))

    rows = []
    for gfunc in gfuncs:
        print(gfunc)
        for c, alpha in paramsCI:
            if alpha==alphas[0]: print(f"\t {c}")
            # k = 0: independent of c, alpha, and gfunc
            for k in memory_sizes:
                if k==0:
                    estimates = beliefs
                else:
                    estimates = simulate(k, c, alpha, gfunc, beliefs, u)
                wocMean_err, ind_err, final_err = metrics(estimates)
                rows.append(pd.DataFrame({
                    "seed": np.repeat(seeds, len(N_evaluate)), 
                    "k": k, "alpha":alpha,"N": np.tile(N_evaluate, n_seeds),
                    "relative_error_wocMean": wocMean_err.ravel(), "relative_error_ind": ind_err.ravel(),
                    "relative_error_final": final_err.ravel(), "gfunc": gfunc, "c": c}))

    errorsCI = pd.concat(rows, ignore_index=True)
    errorsCI.loc[errorsCI["k"] > errorsCI["N"] , [c for c in errorsCI.columns if "relative_error" in c]] = np.nan
    return errorsCI


#%%
if __name__=="__main__":
    errorsCI  = simCI(n,n_seeds,N_evaluate,SIG_CI,T,memory_sizes,paramsCI)
    print("\n" + f"simData/ci{run}{f'_N{n}' if n!=100 else ''}.csv")
    (errorsCI.loc[errorsCI.N==n]).to_csv(f"simData/ci{run}{f'_N{n}' if n!=100 else ''}.csv", index=False)



#%%

# %%

# %%
