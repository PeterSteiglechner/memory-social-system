import numpy as np
from scipy.special import expit as logist
from scipy.special import erf
import scipy.stats as st
import pandas  as  pd
import matplotlib.pyplot as plt

# Variables: 
# E \in R represents noviceness (or inverse expertise), where E=0 is the average expertise level
# G \in R represents tendency to change. When signal==truth, how likely is it that you change something? G=0 means, p to adjust is 50%. G=-1 means p(y==truth)=0.268
# D \in R is difficulty 
# 0<R<=1 is reduction factor of the 'plausibility frame' for new signal creation (compared with acceptance)
# p is propbability to adjust a 


# def p_original(signal, truth, G, D, E): 
#     return logist(G) + (1-logist(G)) * 4 * (norm.cdf((signal - truth)/(D*logist(E)))- 0.5)**2 

# logist = lambda x: (1/(1+np.exp(-x)))

def p(signal, truth, G, D, E): 
    return logist(G) + (1-logist(G)) * (erf((signal - truth)/(D*logist(E)*(2**0.5))))**2 

def new_signal(signal, truth, D, E, R, anchoring=True): 
    if np.isnan(signal):
        loc=truth
        scale=D*logist(E)
    else:
        loc = truth + int(anchoring) * (logist(E)) * (signal - truth)
        scale = R*D*logist(E)
    dist = st.norm(loc=loc, scale=scale)
    return dist.rvs()

agg = lambda x, which: np.median(x) if which=="median" else np.mean(x) # performance aggregtaion



T = 0.2
plt.figure(figsize=(12/2.54, 5/2.54))
ax = plt.axes()
x = np.linspace(-1,1)
for G, D, E in [(-1,3,0), (-1,3,-2), (-1,3,2), (0,3,0), (-1,1,0)]:
    l, = ax.plot(x, 1-p(x, T,  G=G, D=D, E=E), label=f"E={logist(E):.1f}, G={logist(G):.1f}, D={D}", alpha=0.5)
    ax.text(1.05, 1-p(1, T,  G=G, D=D, E=E), f"E={logist(E):.1f}, G={logist(G):.1f}, D={D}" , ha="left", color=l.get_color(), )
ax.set_xlim(-1,1)
ax.set_yticks(np.arange(0,0.81,0.1))
ax.set_xticks(np.arange(-1,1.01,0.5))
ax.grid(axis="y")
#plt.legend(loc="lower left")
ax.set_ylabel("probability to copy\n"+r"$p(x_n|\text{params}\ E,\, D,\, G)$")
ax.set_xlabel(r"social signal $x_n$")
ax.axvline(T, color= "grey")
ax.text(T+0.03, 0.2, f"T={T}")
plt.tight_layout()
plt.savefig(f"figs/pcopy_mayerheck.png", dpi=600)
