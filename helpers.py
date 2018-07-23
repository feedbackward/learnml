
import numpy as np
import math
import os


# Various helpers for our data-generation, and results storage, etc.

def makedir_safe(dirname):
    '''
    A simple utility for making new directories
    after checking that they do not exist.
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def vlnorm(meanlog, sdlog):
    '''
    Variance of the log-Normal distribution.
    '''
    return (math.exp(sdlog**2) - 1) * math.exp((2*meanlog + sdlog**2))

def mlnorm(meanlog, sdlog):
    '''
    Mean of log-Normal distribution.
    '''
    return math.exp((meanlog + sdlog**2/2))


def vnorm(shift, scale):
    '''
    Variance of the Normal distribution.
    '''
    return scale**2

def mnorm(shift, scale):
    '''
    Mean of Normal distribution.
    '''
    return shift


def riskMaker(w, A, b, w_star):
    diff = w - w_star
    quad = diff.T.dot(A.dot(diff).reshape(diff.shape))
    return np.float32(quad) + b**2


def noise_risk(paras):
    
    name = paras["name"]
    
    if name == "lnorm":
        mean_noise = mlnorm(meanlog=paras["meanlog"],
                            sdlog=paras["sdlog"])
        var_noise = vlnorm(meanlog=paras["meanlog"],
                           sdlog=paras["sdlog"])
        
    if name == "norm":
        mean_noise = mnorm(shift=paras["shift"],
                           scale=paras["scale"])
        var_noise = vnorm(shift=paras["meanlog"],
                          scale=paras["sdlog"])

    return (mean_noise, var_noise)


def noise_data(n, paras):
    
    name = paras["name"]
    
    if name == "lnorm":
        mean_noise, var_noise = noise_risk(paras=paras)
        return np.random.lognormal(mean=paras["meanlog"], sigma=paras["sdlog"], size=(n,1))-mean_noise
    
    if name == "norm":
        mean_noise, var_noise = noise_risk(paras=paras)
        return np.random.normal(loc=paras["shift"], scale=paras["scale"], size=(n,1))-mean_noise
