
import numpy as np
import sys


class Redirect:
    '''
    Redirect standard output to a specified file object. This lets us
    use print() to write directly to disk.
    '''
    
    # Set new attribute.
    def __init__(self, o_new):
        self.new = o_new

    # Redirect stdout.
    def __enter__(self):
        self.old = sys.stdout
        sys.stdout = self.new

    # Back to original stdout setting.
    def __exit__(self, *args):
        sys.stdout = self.old


class DataInfo:

    def __init__(self):

        # Shape, path, and data type of train/test in/outputs.
        self.X_tr = {"shape": None,
                     "path": None,
                     "dtype": None}
        self.X_te = {"shape": None,
                     "path": None,
                     "dtype": None}
        self.y_tr = {"shape": None,
                     "path": None,
                     "dtype": None}
        self.y_te = {"shape": None,
                     "path": None,
                     "dtype": None}
        # Desired model name.
        self.mname = None
        # A dictionary for holding anything needed at runtime.
        self.misc = {}

    def __str__(self):

        out = "X_tr: " + str(self.X_tr) + "\n" +\
              "X_te: " + str(self.X_te) + "\n" +\
              "y_tr: " + str(self.y_tr) + "\n" +\
              "y_te: " + str(self.y_te) + "\n" +\
              "mname: " + str(self.mname) + "\n" +\
              "misc: " + str(self.misc)
        
        return out

        
class Data:
    '''
    A general data class, assumed to be given a data info
    object with dictionaries containing the path and shape
    information of data to be used in the final routine.
    '''

    def __init__(self, dinfo):

        # Inputs, training.
        myd = dinfo.X_tr
        if myd:
            with open(myd["path"], mode="br") as f:
                myar = np.fromfile(file=f, dtype=myd["dtype"])
                self.X_tr = myar.reshape(myd["shape"])
        else:
            self.X_tr = None

        # Inputs, testing.
        myd = dinfo.X_te
        if myd:
            with open(myd["path"], mode="br") as f:
                myar = np.fromfile(file=f, dtype=myd["dtype"])
                self.X_te = myar.reshape(myd["shape"])
        else:
            self.X_te = None
            
        # Outputs, training.
        myd = dinfo.y_tr
        if myd:
            with open(myd["path"], mode="br") as f:
                myar = np.fromfile(file=f, dtype=myd["dtype"])
                self.y_tr = myar.reshape(myd["shape"])
        else:
            self.y_tr = None

        # Outputs, testing.
        myd = dinfo.y_te
        if myd:
            with open(myd["path"], mode="br") as f:
                myar = np.fromfile(file=f, dtype=myd["dtype"])
                self.y_te = myar.reshape(myd["shape"])
        else:
            self.y_te = None

    def __str__(self):

        val = (str(self.X_tr.shape) if self.X_tr is not None else str(None))
        s_Xtr = "X_tr:" + val
        val = (str(self.X_te.shape) if self.X_te is not None else str(None))
        s_Xte = "X_te:" + val
        val = (str(self.y_tr.shape) if self.y_tr is not None else str(None))
        s_ytr = "y_tr:" + val
        val = (str(self.y_te.shape) if self.y_te is not None else str(None))
        s_yte = "y_te:" + val
        return s_Xtr + "\n" + s_Xte + "\n" + s_ytr + "\n" + s_yte + "\n"


class Itin:
    '''
    A simple class which is instantiated using a pair of
    dictionaries.
    '''

    def __init__(self, exp, mth):
        self.exp = exp
        self.mth = mth
