
import support.classes as classes
import os
import pickle
import numpy as np
import csv
import math

DATA_PATH = os.path.join(os.path.expanduser('~'), "learnml/data")

def prep(s, skipread=False):
    '''
    Takes a string with data set name, and runs the proper setup.
    '''

    print("Preparing data (s =", s, ")...")

    if s == "toyReg":
        toread = os.path.join("data", s, "info.dat")
        if skipread:
            with open(toread, mode="br") as f:
                dinfo = pickle.load(f)
            return dinfo
        else:
            return toyReg()

    if s == "toyClass":
        toread = os.path.join("data", s, "info.dat")
        if skipread:
            with open(toread, mode="br") as f:
                dinfo = pickle.load(f)
            return dinfo
        else:
            return toyClass()

    if s == "MNIST":
        toread = os.path.join("data", s, "info.dat")
        if skipread:
            with open(toread, mode="br") as f:
                dinfo = pickle.load(f)
            return dinfo
        else:
            return MNIST()
    
    if s == "quantum":
        toread = os.path.join("data", s, "info.dat")
        if skipread:
            with open(toread, mode="br") as f:
                dinfo = pickle.load(f)
            return dinfo
        else:
            return quantum()

    if s == "NoisyOpt_isoBig":
        toread = os.path.join("data", s, "info.dat")
        if skipread:
            with open(toread, mode="br") as f:
                dinfo = pickle.load(f)
            return dinfo
        else:
            return NoisyOpt_isoBig()

    if s == "NoisyOpt_isoSmall":
        toread = os.path.join("data", s, "info.dat")
        if skipread:
            with open(toread, mode="br") as f:
                dinfo = pickle.load(f)
            return dinfo
        else:
            return NoisyOpt_isoSmall()



def load(dinfo):
    '''
    Given the info (path to binary, shape) about a particular data set,
    load relevant training and testing data sets.    
    '''
    print("Reading data...")
    return classes.Data(dinfo)



def MNIST():
    '''
    Data preparation function, specific to the MNIST handwritten
    digits data set. 
    URL: http://yann.lecun.com/exdb/mnist/
    '''
    dataset = "MNIST"
    dinfo = classes.DataInfo()
    dinfo.mname = "LgstReg" # hard-coded model name.

    print("Preparation (", dataset, ")...")
    print("Inputs (training)...")
    toread = os.path.join(DATA_PATH,
                          dataset,
                          "train-images-idx3-ubyte")
    towrite = os.path.join("data", dataset, ("X_tr" + ".dat"))
    with open(toread, mode="rb") as f_bin:
    
        f_bin.seek(4)
        b = f_bin.read(4)
        n = int.from_bytes(b, byteorder="big")
        b = f_bin.read(4)
        d_rows = int.from_bytes(b, byteorder="big")
        b = f_bin.read(4)
        d_cols = int.from_bytes(b, byteorder="big")
        d = d_rows * d_cols

        with open(towrite, mode="bw") as g_bin:
    
            bytes_left = n * d
            idx = 0
            data_arr = np.empty( (n*d), dtype=np.uint8 )
            while bytes_left > 0:
                b = f_bin.read(1)
                data_arr[idx] = np.uint8(int.from_bytes(b, byteorder="big"))
                bytes_left -= 1
                idx += 1
            
            data_arr.tofile(g_bin)
            
    dinfo.X_tr["shape"] = (n,d)
    dinfo.X_tr["path"] = towrite
    dinfo.X_tr["dtype"] = np.uint8

    # --------------------------- #

    print("Inputs (testing)...")
    toread = os.path.join(DATA_PATH,
                          dataset,
                          "t10k-images-idx3-ubyte")
    towrite = os.path.join("data", dataset, ("X_te" + ".dat"))
    with open(toread, mode="rb") as f_bin:
    
        f_bin.seek(4)
        b = f_bin.read(4)
        n = int.from_bytes(b, byteorder="big")
        b = f_bin.read(4)
        d_rows = int.from_bytes(b, byteorder="big")
        b = f_bin.read(4)
        d_cols = int.from_bytes(b, byteorder="big")
        d = d_rows * d_cols

        with open(towrite, mode="bw") as g_bin:
        
            bytes_left = n * d
            idx = 0
            data_arr = np.empty( (n*d), dtype=np.uint8 )
            while bytes_left > 0:
                b = f_bin.read(1)
                data_arr[idx] = np.uint8(int.from_bytes(b, byteorder="big"))
                bytes_left -= 1
                idx += 1

            data_arr.tofile(g_bin)
            
    dinfo.X_te["shape"] = (n,d)
    dinfo.X_te["path"] = towrite
    dinfo.X_te["dtype"] = np.uint8

    # --------------------------- #

    print("Outputs (training)...")
    toread = os.path.join(DATA_PATH,
                          dataset,
                          "train-labels-idx1-ubyte")
    towrite = os.path.join("data", dataset, ("y_tr" + ".dat"))
    with open(toread, mode="rb") as f_bin:
    
        f_bin.seek(4)
        b = f_bin.read(4)
        n = int.from_bytes(b, byteorder="big")
        d = 1
        
        with open(towrite, mode="bw") as g_bin:
        
            bytes_left = n * d
            idx = 0
            data_arr = np.empty( (n*d), dtype=np.uint8 )
            while bytes_left > 0:
                b = f_bin.read(1)
                data_arr[idx] = np.uint8(int.from_bytes(b, byteorder="big"))
                bytes_left -= 1
                idx += 1

            data_arr.tofile(g_bin)

    dinfo.y_tr["shape"] = (n,d)
    dinfo.y_tr["path"] = towrite
    dinfo.y_tr["dtype"] = np.uint8


    # --------------------------- #

    print("Outputs (testing)...")
    toread = os.path.join(DATA_PATH,
                          dataset,
                          "t10k-labels-idx1-ubyte")
    towrite = os.path.join("data", dataset, ("y_te" + ".dat"))
    with open(toread, mode="rb") as f_bin:
    
        f_bin.seek(4)
        b = f_bin.read(4)
        n = int.from_bytes(b, byteorder="big")
        d = 1
    
        with open(towrite, mode="bw") as g_bin:
        
            bytes_left = n * d
            idx = 0
            data_arr = np.empty( (n*d), dtype=np.uint8 )
            while bytes_left > 0:
                b = f_bin.read(1)
                data_arr[idx] = np.uint8(int.from_bytes(b, byteorder="big"))
                bytes_left -= 1
                idx += 1
            
            data_arr.tofile(g_bin)

    dinfo.y_te["shape"] = (n,d)
    dinfo.y_te["path"] = towrite
    dinfo.y_te["dtype"] = np.uint8


    # Save the dinfo dictionary for future use (so we don't have to read
    # the original data every time).
    towrite = os.path.join("data", dataset, "info.dat")
    with open(towrite, mode="wb") as f:
        pickle.dump(dinfo, f)

    # Finally, return the dinfo dict.
    return dinfo



def NoisyOpt_isoBig():
    '''
    Data preparation routine for "noisy optimization" demo,
    where inputs are generated from a linear model with
    additive noise, such that the expected value of the
    squared loss of each is a known (and thus computable) risk
    function. The "iso**Big**" part means that the sample is
    rather large (and likely will be sub-sampled in practice).
    
    '''
    dataset = "NoisyOpt_isoBig"
    dinfo = classes.DataInfo()

    dinfo.X_te = None # since have risk oracle, only use training data.
    dinfo.y_te = None

    n = 50 # training set size
    d = 2 # number of inputs
    sigma_X = 1 # magnitude of input stdev
    sigma_noise = 3 # magnitude of noise stdev
    delta = sigma_noise * 5 # the amount of displacement of initial value
    dinfo.mname = "NoisyOpt" # hard-coded model name.
    dinfo.misc["cov_X"] = (sigma_X**2) * np.eye(d) # cov mtx of inputs
    dinfo.misc["sigma_noise"] = sigma_noise
    dinfo.misc["nsub"] = 5 # the size of random sub-samples to be used.

    # Hand-prepared data, used below.
    w_true = np.array([3.141592, 1.414214]).reshape((d,1))
    X_tr = np.random.normal(loc=0.0,
                            scale=sigma_X,
                            size=n*d).reshape((n,d))
    noise_tr = np.random.normal(loc=0.0,
                                scale=sigma_noise,
                                size=n).reshape((n,1))
    dinfo.misc["w_true"] = w_true # store the true model paras.
    dinfo.misc["w_init"] = w_true + delta # store a fixed initial value

    # Inputs
    towrite = os.path.join("data", dataset, ("X_tr" + ".dat"))
    data_arr = X_tr
    dinfo.X_tr["shape"] = data_arr.shape
    dinfo.X_tr["path"] = towrite
    dinfo.X_tr["dtype"] = data_arr.dtype
    with open(towrite, mode="bw") as g_bin:
        data_arr.tofile(g_bin)

    # Outputs
    towrite = os.path.join("data", dataset, ("y_tr" + ".dat"))
    data_arr = (np.dot(X_tr,w_true)+noise_tr).reshape((X_tr.shape[0],1))
    dinfo.y_tr["shape"] = data_arr.shape
    dinfo.y_tr["path"] = towrite
    dinfo.y_tr["dtype"] = data_arr.dtype
    with open(towrite, mode="bw") as g_bin:
        data_arr.tofile(g_bin)

    # Save the dinfo dictionary for future use (so we don't have to read
    # the original data every time).
    towrite = os.path.join("data", dataset, "info.dat")
    with open(towrite, mode="wb") as f:
        pickle.dump(dinfo, f)

    # Finally, return the dinfo dict.
    return dinfo


def NoisyOpt_isoSmall():
    '''
    Data preparation routine for "noisy optimization" demo,
    where inputs are generated from a linear model with
    additive noise, such that the expected value of the
    squared loss of each is a known (and thus computable) risk
    function. The "iso**Small**" part means that the sample is
    rather small, and will be used as a batch (no sub-sampling).
    
    '''
    dataset = "NoisyOpt_isoSmall"
    dinfo = classes.DataInfo()

    dinfo.X_te = None # since have risk oracle, only use training data.
    dinfo.y_te = None

    n = 15 # training set size
    d = 2 # number of inputs
    sigma_X = 1 # magnitude of input stdev
    sigma_noise = 3 # magnitude of noise stdev
    delta = sigma_noise * 5 # the amount of displacement of initial value
    dinfo.mname = "NoisyOpt" # hard-coded model name.
    dinfo.misc["cov_X"] = (sigma_X**2) * np.eye(d) # cov mtx of inputs
    dinfo.misc["sigma_noise"] = sigma_noise
    dinfo.misc["nsub"] = n # no sub-sampling, use whole batch.
    
    # Hand-prepared data, used below.
    w_true = np.array([3.141592, 1.414214]).reshape((d,1))
    X_tr = np.random.normal(loc=0.0,
                            scale=sigma_X,
                            size=n*d).reshape((n,d))
    noise_tr = np.random.normal(loc=0.0,
                                scale=sigma_noise,
                                size=n).reshape((n,1))
    dinfo.misc["w_true"] = w_true # store the true model paras.
    dinfo.misc["w_init"] = w_true + delta # store a fixed initial value
    
    # Inputs
    towrite = os.path.join("data", dataset, ("X_tr" + ".dat"))
    data_arr = X_tr
    dinfo.X_tr["shape"] = data_arr.shape
    dinfo.X_tr["path"] = towrite
    dinfo.X_tr["dtype"] = data_arr.dtype
    with open(towrite, mode="bw") as g_bin:
        data_arr.tofile(g_bin)

    # Outputs
    towrite = os.path.join("data", dataset, ("y_tr" + ".dat"))
    data_arr = (np.dot(X_tr,w_true)+noise_tr).reshape((X_tr.shape[0],1))
    dinfo.y_tr["shape"] = data_arr.shape
    dinfo.y_tr["path"] = towrite
    dinfo.y_tr["dtype"] = data_arr.dtype
    with open(towrite, mode="bw") as g_bin:
        data_arr.tofile(g_bin)

    # Save the dinfo dictionary for future use (so we don't have to read
    # the original data every time).
    towrite = os.path.join("data", dataset, "info.dat")
    with open(towrite, mode="wb") as f:
        pickle.dump(dinfo, f)

    # Finally, return the dinfo dict.
    return dinfo


def NoisyOpt_SmallSparse():
    '''
    A small simulated data set based on a linear regression model
    with additive noise and a sparse underlying model vector. This
    comes from the Elements of Statistical Learning (ESL2) text,
    in Figure 3.6 on page 59 (with n=300), and again (with n=100)
    on page 78 in Figure 3.16.

    NOTE: since most of the methods we shall be using involve a
    centering and standardizing of the data, we elect to do this
    here, at the time of generation. That is, the sample given
    has EMPIRICAL mean of zero and EMPIRICAL variance of one.
    '''
    dataset = "NoisyOpt_SmallSparse"
    dinfo = classes.DataInfo()

    n = 100 # training set size
    d = 31 # number of inputs
    d0 = 10 # number of *active* inputs
    sigma_X = 1.0 # unit variance
    corr = 0.85 # pairwise correlation coefficient
    sigma_noise = math.sqrt(6.25) # stdev of additive noise
    sigma_weights = math.sqrt(0.4) # stdev of randomly generated weights
    dinfo.misc["sigma_noise"] = sigma_noise
    
    cov_X = np.zeros(d*d).reshape((d,d)) + corr # prepare cov mtx
    np.fill_diagonal(cov_X, sigma_X)
    dinfo.misc["cov_X"] = cov_X # cov mtx of inputs

    w_true = np.zeros(d).reshape((d,1)) # prepare the model vec
    idx_on = np.random.choice(d, size=d0, replace=False)
    w_true[idx_on,:] = np.random.normal(loc=0.0,
                                        scale=sigma_weights,
                                        size=d0).reshape((d0,1))
    dinfo.misc["w_true"] = w_true # store the true model paras.

    # Generate the actual data.
    X_tr = np.random.multivariate_normal(mean=np.zeros(d), cov=cov_X, size=n)
    noise_tr = np.random.normal(loc=0.0,
                                scale=sigma_noise,
                                size=n).reshape((n,1))
    
    y_tr = np.dot(X_tr,w_true) + noise_tr

    # Standardize the inputs to have unit (empirical) variance.
    X_tr = (X_tr-np.mean(X_tr,axis=0)) / np.sqrt(np.var(X_tr,axis=0))
    
    # Remaining parameters for the model object.
    delta = sigma_noise * 1 # the amount of displacement of initial value
    dinfo.misc["w_init"] = w_true + delta # store a fixed initial value
    dinfo.mname = "NoisyOpt" # hard-coded model name.
    dinfo.misc["nsub"] = n # no sub-sampling, use whole batch.
    
    # Inputs
    towrite = os.path.join("data", dataset, ("X_tr" + ".dat"))
    data_arr = X_tr
    dinfo.X_tr["shape"] = data_arr.shape
    dinfo.X_tr["path"] = towrite
    dinfo.X_tr["dtype"] = data_arr.dtype
    with open(towrite, mode="bw") as g_bin:
        data_arr.tofile(g_bin)

    # Outputs
    towrite = os.path.join("data", dataset, ("y_tr" + ".dat"))
    data_arr = (np.dot(X_tr,w_true)+noise_tr).reshape((X_tr.shape[0],1))
    dinfo.y_tr["shape"] = data_arr.shape
    dinfo.y_tr["path"] = towrite
    dinfo.y_tr["dtype"] = data_arr.dtype
    with open(towrite, mode="bw") as g_bin:
        data_arr.tofile(g_bin)

    dinfo.X_te = None # Have an oracle
    dinfo.y_te = None

    # Save the dinfo dictionary for future use (so we don't have to read
    # the original data every time).
    towrite = os.path.join("data", dataset, "info.dat")
    with open(towrite, mode="wb") as f:
        pickle.dump(dinfo, f)

    # Finally, return the dinfo dict.
    return dinfo


    
