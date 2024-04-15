# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Poly case

# +
# Originally from Feda in DASS:
# https://github.com/equinor/dass/blob/753fbae317ed6e025cc3a745b3e6831c583bd746/notebooks/Poly.py)
# -

# ### Case 1 - ES uniform prior no mapping to normal dist

# +
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
rng = np.random.default_rng(12345)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams.update({"font.size": 10})
from ipywidgets import interact
import ipywidgets as widgets

from p_tqdm import p_map

from scipy.ndimage import gaussian_filter

from scipy.stats import uniform
from scipy.stats import norm

#import iterative_ensemble_smoother as 

# +

from dass import pde, utils, analysis, taper
# -

N = 1000


def poly(a, b, c, x):
    return a * x**2 + b * x + c


# +
a_t = 0.5
b_t = 1.0
c_t = 3.0

x_observations = [0, 2, 4, 6, 8]
# observations = [
#     (
#         poly(a_t, b_t, c_t, x) + rng.normal(loc=0, scale=0.2 * poly(a_t, b_t, c_t, x)),
#         0.2 * poly(a_t, b_t, c_t, x),
#         x,
#     )
#     for x in x_observations
# ]

# case with  std = 0.001
observations = [
    (
        poly(a_t, b_t, c_t, x) + rng.normal(loc=0, scale=0.001),
        0.001,
        x,
    )
    for x in x_observations
]

d = pd.DataFrame(observations, columns=["value", "sd", "x"])

d = d.set_index("x")

m = d.shape[0]
# -

fig, ax = plt.subplots()
x_plot = np.linspace(0, 10, 50)
ax.set_title("Truth and noisy observations")
ax.set_xlabel("Time step")
ax.set_ylabel("Response")
ax.plot(x_plot, poly(a_t, b_t, c_t, x_plot))
ax.plot(d.index.get_level_values("x"), d.value.values, "o")
ax.grid()

# +
# Assume diagonal ensemble covariance matrix for the measurement perturbations.
Cdd = np.diag(d.sd.values**2)

E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=N).T
E = E - E.mean(axis=1, keepdims=True)
assert E.shape == (m, N)

D = np.ones((m, N)) * d.value.values.reshape(-1, 1) + E

# +
# coeff_a = rng.normal(0, 1, size=N)
# coeff_b = rng.normal(0, 1, size=N)
# coeff_c = rng.normal(0, 1, size=N)

#uniform dist to repreduce tutorial in ERT webpage
coeff_a = rng.uniform(0, 1, size=N)
coeff_b = rng.uniform(0, 2, size=N)
coeff_c = rng.uniform(0, 5, size=N)
# -

A = np.concatenate(
    (coeff_a.reshape(-1, 1), coeff_b.reshape(-1, 1), coeff_c.reshape(-1, 1)), axis=1
).T

fwd_runs = p_map(
    poly,
    coeff_a,
    coeff_b,
    coeff_c,
    [np.arange(max(x_observations) + 1)] * N,
    desc=f"Running forward model.",
)

# +
Y = np.array(
    [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in fwd_runs]
).T

assert Y.shape == (
    m,
    N,
), "Measured responses must be a matrix with dimensions (number of observations x number of realisations)"
# -

X = analysis.ES(Y, D, Cdd)
A_ES = A @ X

# +
fig, ax = plt.subplots(nrows=3, ncols=2)
fig.set_size_inches(10, 10)

ax[0, 0].set_title("a - prior")
ax[0, 0].hist(A[0, :])

ax[0, 1].set_title("a - posterior")
ax[0, 1].hist(A_ES[0, :])
ax[0, 1].axvline(a_t, color="k", linestyle="--", label="truth")
ax[0, 1].legend()
# ax[0, 1].set_xlim([0.4, 0.6])

ax[1, 0].set_title("b - prior")
ax[1, 0].hist(A[1, :])

ax[1, 1].set_title("b - posterior")
ax[1, 1].hist(A_ES[1, :])
ax[1, 1].axvline(b_t, color="k", linestyle="--", label="truth")
ax[1, 1].legend()
# ax[1, 1].set_xlim([0.90, 1.1])

ax[2, 0].set_title("c - prior")
ax[2, 0].hist(A[2, :])

ax[2, 1].set_title("c - posterior")
ax[2, 1].hist(A_ES[2, :])
ax[2, 1].axvline(c_t, color="k", linestyle="--", label="truth")
ax[2, 1].legend()
# ax[2, 1].set_xlim([2.9, 3.1])

fig.tight_layout()

# +

# %%
fig, ax = plt.subplots()
x_plot = np.linspace(0, 10, 50)
ax.set_title("Obs, Prior and Post")
ax.set_xlabel("Time step")
ax.set_ylabel("Response")
[ax.plot(x_plot, poly(A[0,i], A[1,i], A[2,i], x_plot),color= "gray", alpha=0.05) for i in range(N)] 
[ax.plot(x_plot, poly(A_ES[0,i], A_ES[1,i], A_ES[2,i], x_plot),color= "blue") for i in range(N)] 
ax.plot(x_plot, poly(a_t, b_t, c_t, x_plot))
ax.plot(d.index.get_level_values("x"), d.value.values, "o")

ax.grid()
# -

# ### Case 2 -  ES with prior transform to standard normal before updating

# +
# to be cleaned and refactored 
# -

N = 1000


def poly(a, b, c, x):
    return a * x**2 + b * x + c


# +
a_t = 0.5
b_t = 1.0
c_t = 3.0

x_observations = [0, 2, 4, 6, 8]
# observations = [
#     (
#         poly(a_t, b_t, c_t, x) + rng.normal(loc=0, scale=0.2 * poly(a_t, b_t, c_t, x)),
#         0.2 * poly(a_t, b_t, c_t, x),
#         x,
#     )
#     for x in x_observations
# ]

# case with  std = 0.001
observations = [
    (
        poly(a_t, b_t, c_t, x) + rng.normal(loc=0, scale=0.001),
        0.001,
        x,
    )
    for x in x_observations
]

d = pd.DataFrame(observations, columns=["value", "sd", "x"])

d = d.set_index("x")

m = d.shape[0]
# -

fig, ax = plt.subplots()
x_plot = np.linspace(0, 10, 50)
ax.set_title("Truth and noisy observations")
ax.set_xlabel("Time step")
ax.set_ylabel("Response")
ax.plot(x_plot, poly(a_t, b_t, c_t, x_plot))
ax.plot(d.index.get_level_values("x"), d.value.values, "o")
ax.grid()

# +
# Assume diagonal ensemble covariance matrix for the measurement perturbations.
Cdd = np.diag(d.sd.values**2)

E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=N).T
E = E - E.mean(axis=1, keepdims=True)
assert E.shape == (m, N)

D = np.ones((m, N)) * d.value.values.reshape(-1, 1) + E


# +
#define functions to map uniform to normal and normal to uniform distibutions 
def unifor_to_normal(vector,loc,scale,mu=0,std=1):
    return(np.array([norm.ppf(uniform.cdf(i, loc=loc, scale=scale), loc=mu, scale=std) for i in vector]))

def normal_to_uniform(vector,loc,scale,mu=0,std=1):
    return(np.array([uniform.ppf(norm.cdf(i, loc=mu, scale=std), loc=loc, scale=scale) for i in vector]))



# +
# coeff_a = rng.normal(0, 1, size=N)
# coeff_b = rng.normal(0, 1, size=N)
# coeff_c = rng.normal(0, 1, size=N)

#uniform dist to repreduce tutorial in ERT webpage
coeff_a = unifor_to_normal(rng.uniform(0, 1, size=N),loc = 0,scale = 1)
coeff_b = unifor_to_normal(rng.uniform(0, 2, size=N),loc = 0,scale = 2)
coeff_c = unifor_to_normal(rng.uniform(0, 5, size=N),loc = 0,scale = 5)
# -

A = np.concatenate(
    (coeff_a.reshape(-1, 1), coeff_b.reshape(-1, 1), coeff_c.reshape(-1, 1)), axis=1
).T

fwd_runs = p_map(
    poly,
    normal_to_uniform(coeff_a,0,1,mu=0,std=1),
    normal_to_uniform(coeff_b,0,2,mu=0,std=1),
    normal_to_uniform(coeff_c,0,5,mu=0,std=1),
    [np.arange(max(x_observations) + 1)] * N,
    desc=f"Running forward model.",
)

# +
Y = np.array(
    [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in fwd_runs]
).T

assert Y.shape == (
    m,
    N,
), "Measured responses must be a matrix with dimensions (number of observations x number of realisations)"
# -

X = analysis.ES(Y, D, Cdd)
A_ES = A @ X

# +
#transforming back to uniform dist

A[0,:] = normal_to_uniform(A[0,:],0,1,mu=0,std=1)
A[1,:] = normal_to_uniform(A[1,:],0,2,mu=0,std=1)
A[2,:] = normal_to_uniform(A[2,:],0,5,mu=0,std=1)
A_ES[0,:] = normal_to_uniform(A_ES[0,:],0,1,mu=0,std=1)
A_ES[1,:] = normal_to_uniform(A_ES[1,:],0,2,mu=0,std=1)
A_ES[2,:] = normal_to_uniform(A_ES[2,:],0,5,mu=0,std=1)

# +
fig, ax = plt.subplots(nrows=3, ncols=2)
fig.set_size_inches(10, 10)

ax[0, 0].set_title("a - prior")
ax[0, 0].hist(A[0, :])

ax[0, 1].set_title("a - posterior")
ax[0, 1].hist(A_ES[0, :])
ax[0, 1].axvline(a_t, color="k", linestyle="--", label="truth")
ax[0, 1].legend()
# ax[0, 1].set_xlim([0.4, 0.6])

ax[1, 0].set_title("b - prior")
ax[1, 0].hist(A[1, :])

ax[1, 1].set_title("b - posterior")
ax[1, 1].hist(A_ES[1, :])
ax[1, 1].axvline(b_t, color="k", linestyle="--", label="truth")
ax[1, 1].legend()
# ax[1, 1].set_xlim([0.90, 1.1])

ax[2, 0].set_title("c - prior")
ax[2, 0].hist(A[2, :])

ax[2, 1].set_title("c - posterior")
ax[2, 1].hist(A_ES[2, :])
ax[2, 1].axvline(c_t, color="k", linestyle="--", label="truth")
ax[2, 1].legend()
# ax[2, 1].set_xlim([2.9, 3.1])

fig.tight_layout()

# +

# %%
fig, ax = plt.subplots()
x_plot = np.linspace(0, 10, 50)
ax.set_title("Obs, Prior and Post")
ax.set_xlabel("Time step")
ax.set_ylabel("Response")
[ax.plot(x_plot, poly(A[0,i], A[1,i], A[2,i], x_plot),color= "gray", alpha=0.05) for i in range(N)] 
[ax.plot(x_plot, poly(A_ES[0,i], A_ES[1,i], A_ES[2,i], x_plot),color= "blue") for i in range(N)] 
ax.plot(x_plot, poly(a_t, b_t, c_t, x_plot))
ax.plot(d.index.get_level_values("x"), d.value.values, "o")

ax.grid()
# -

# ### ESMDA  with prior transform to standard normal before updating

# +
# to be cleaned and (very) refactored 
# -

N = 1000


def poly(a, b, c, x):
    return a * x**2 + b * x + c


# +
a_t = 0.5
b_t = 1.0
c_t = 3.0

x_observations = [0, 2, 4, 6, 8]
# observations = [
#     (
#         poly(a_t, b_t, c_t, x) + rng.normal(loc=0, scale=0.2 * poly(a_t, b_t, c_t, x)),
#         0.2 * poly(a_t, b_t, c_t, x),
#         x,
#     )
#     for x in x_observations
# ]

# case with  std = 0.001
observations = [
    (
        poly(a_t, b_t, c_t, x) + rng.normal(loc=0, scale=0.001),
        0.001,
        x,
    )
    for x in x_observations
]

d = pd.DataFrame(observations, columns=["value", "sd", "x"])

d = d.set_index("x")

m = d.shape[0]
# -

fig, ax = plt.subplots()
x_plot = np.linspace(0, 10, 50)
ax.set_title("Truth and noisy observations")
ax.set_xlabel("Time step")
ax.set_ylabel("Response")
ax.plot(x_plot, poly(a_t, b_t, c_t, x_plot))
ax.plot(d.index.get_level_values("x"), d.value.values, "o")
ax.grid()

# +
# Assume diagonal ensemble covariance matrix for the measurement perturbations.
Cdd = np.diag(d.sd.values**2)
Cdd = np.diag(d.sd.values**2)*4 #emulate esmda 4itr with constant alpha = 4

E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=N).T
E = E - E.mean(axis=1, keepdims=True)
assert E.shape == (m, N)

D = np.ones((m, N)) * d.value.values.reshape(-1, 1) + E


# +
# coeff_a = rng.normal(0, 1, size=N)
# coeff_b = rng.normal(0, 1, size=N)
# coeff_c = rng.normal(0, 1, size=N)

#define functions to map uniform to normal and normal to uniform distibutions 
def unifor_to_normal(vector,loc,scale,mu=0,std=1):
    return(np.array([norm.ppf(uniform.cdf(i, loc=loc, scale=scale), loc=mu, scale=std) for i in vector]))

def normal_to_uniform(vector,loc,scale,mu=0,std=1):
    return(np.array([uniform.ppf(norm.cdf(i, loc=mu, scale=std), loc=loc, scale=scale) for i in vector]))


#uniform dist to repreduce tutorial in ERT webpage
coeff_a = unifor_to_normal(rng.uniform(0, 1, size=N),loc = 0,scale = 1)
coeff_b = unifor_to_normal(rng.uniform(0, 2, size=N),loc = 0,scale = 2)
coeff_c = unifor_to_normal(rng.uniform(0, 5, size=N),loc = 0,scale = 5)
# -

A = np.concatenate(
    (coeff_a.reshape(-1, 1), coeff_b.reshape(-1, 1), coeff_c.reshape(-1, 1)), axis=1
).T

fwd_runs = p_map(
    poly,
    normal_to_uniform(coeff_a,0,1,mu=0,std=1),
    normal_to_uniform(coeff_b,0,2,mu=0,std=1),
    normal_to_uniform(coeff_c,0,5,mu=0,std=1),
    [np.arange(max(x_observations) + 1)] * N,
    desc=f"Running forward model.",
)

# +
Y = np.array(
    [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in fwd_runs]
).T

assert Y.shape == (
    m,
    N,
), "Measured responses must be a matrix with dimensions (number of observations x number of realisations)"

# +
X = analysis.ES(Y, D, Cdd)
A_ES = A @ X



# +
#iter 2

Cdd = np.diag(d.sd.values**2)*4 #emulate esmda 4itr with constant alpha = 4

E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=N).T
E = E - E.mean(axis=1, keepdims=True)
assert E.shape == (m, N)

D = np.ones((m, N)) * d.value.values.reshape(-1, 1) + E


fwd_runs = p_map(
    poly,
    normal_to_uniform(A_ES[0,:],0,1,mu=0,std=1), 
    normal_to_uniform(A_ES[1,:],0,2,mu=0,std=1),
    normal_to_uniform(A_ES[2,:],0,5,mu=0,std=1),
    [np.arange(max(x_observations) + 1)] * N,
    desc=f"Running forward model.",
)

Y = np.array(
    [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in fwd_runs]
).T

assert Y.shape == (
    m,
    N,
), "Measured responses must be a matrix with dimensions (number of observations x number of realisations)"

X = analysis.ES(Y, D, Cdd)
A_ES = A_ES @ X

#iter 3
Cdd = np.diag(d.sd.values**2)*4 #emulate esmda 4itr with constant alpha = 4

E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=N).T
E = E - E.mean(axis=1, keepdims=True)
assert E.shape == (m, N)

D = np.ones((m, N)) * d.value.values.reshape(-1, 1) + E


fwd_runs = p_map(
    poly,
    normal_to_uniform(A_ES[0,:],0,1,mu=0,std=1), 
    normal_to_uniform(A_ES[1,:],0,2,mu=0,std=1),
    normal_to_uniform(A_ES[2,:],0,5,mu=0,std=1),
    [np.arange(max(x_observations) + 1)] * N,
    desc=f"Running forward model.",
)

Y = np.array(
    [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in fwd_runs]
).T

assert Y.shape == (
    m,
    N,
), "Measured responses must be a matrix with dimensions (number of observations x number of realisations)"

X = analysis.ES(Y, D, Cdd)
A_ES = A_ES @ X


#iter 4
Cdd = np.diag(d.sd.values**2)*4 #emulate esmda 4itr with constant alpha = 4

E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=N).T
E = E - E.mean(axis=1, keepdims=True)
assert E.shape == (m, N)

D = np.ones((m, N)) * d.value.values.reshape(-1, 1) + E


fwd_runs = p_map(
    poly,
    normal_to_uniform(A_ES[0,:],0,1,mu=0,std=1), 
    normal_to_uniform(A_ES[1,:],0,2,mu=0,std=1),
    normal_to_uniform(A_ES[2,:],0,5,mu=0,std=1),
    [np.arange(max(x_observations) + 1)] * N,
    desc=f"Running forward model.",
)

Y = np.array(
    [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in fwd_runs]
).T

assert Y.shape == (
    m,
    N,
), "Measured responses must be a matrix with dimensions (number of observations x number of realisations)"

X = analysis.ES(Y, D, Cdd)
A_ES = A_ES @ X


# +
#transforming back to uniform dist

A[0,:] = normal_to_uniform(A[0,:],0,1,mu=0,std=1)
A[1,:] = normal_to_uniform(A[1,:],0,2,mu=0,std=1)
A[2,:] = normal_to_uniform(A[2,:],0,5,mu=0,std=1)
A_ES[0,:] = normal_to_uniform(A_ES[0,:],0,1,mu=0,std=1)
A_ES[1,:] = normal_to_uniform(A_ES[1,:],0,2,mu=0,std=1)
A_ES[2,:] = normal_to_uniform(A_ES[2,:],0,5,mu=0,std=1)

# +
fig, ax = plt.subplots(nrows=3, ncols=2)
fig.set_size_inches(10, 10)

ax[0, 0].set_title("a - prior")
ax[0, 0].hist(A[0, :])

ax[0, 1].set_title("a - posterior")
ax[0, 1].hist(A_ES[0, :])
ax[0, 1].axvline(a_t, color="k", linestyle="--", label="truth")
ax[0, 1].legend()
# ax[0, 1].set_xlim([0.4, 0.6])

ax[1, 0].set_title("b - prior")
ax[1, 0].hist(A[1, :])

ax[1, 1].set_title("b - posterior")
ax[1, 1].hist(A_ES[1, :])
ax[1, 1].axvline(b_t, color="k", linestyle="--", label="truth")
ax[1, 1].legend()
# ax[1, 1].set_xlim([0.90, 1.1])

ax[2, 0].set_title("c - prior")
ax[2, 0].hist(A[2, :])

ax[2, 1].set_title("c - posterior")
ax[2, 1].hist(A_ES[2, :])
ax[2, 1].axvline(c_t, color="k", linestyle="--", label="truth")
ax[2, 1].legend()
# ax[2, 1].set_xlim([2.9, 3.1])

fig.tight_layout()

# +

# %%
fig, ax = plt.subplots()
x_plot = np.linspace(0, 10, 50)
ax.set_title("Obs, Prior and Post")
ax.set_xlabel("Time step")
ax.set_ylabel("Response")
[ax.plot(x_plot, poly(A[0,i], A[1,i], A[2,i], x_plot),color= "gray", alpha=0.05) for i in range(N)] 
[ax.plot(x_plot, poly(A_ES[0,i], A_ES[1,i], A_ES[2,i], x_plot),color= "blue") for i in range(N)] 
ax.plot(x_plot, poly(a_t, b_t, c_t, x_plot))
ax.plot(d.index.get_level_values("x"), d.value.values, "o")

ax.grid()

# +
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

N = 10
x = np.random.randn(N)
y = np.random.randn(N)

plt.hist(x)
plt.show()

plt.hist(y)
plt.show()




# Fit with polyfit
b, m = polyfit(x, y, 1)

#import numpy as np

plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')
plt.show()

# +



from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

N = 10000
x = np.random.randn(N)
y = np.random.randn(N)

fig, ax = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(20, 8)

ax[0].hist(x)
# plt.show()

ax[1].hist(y)
# plt.show()




# Fit with polyfit
b, m = polyfit(x, y, 1)

#import numpy as np

ax[2].plot(x, y, '.')
ax[2].plot(x, b + m * x, '-')
# plt.show()
fig.tight_layout()
# -




