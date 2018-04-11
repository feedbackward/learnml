
import math
import numpy as np


class Algo_trivial:

    '''
    An iterator which does nothing special at all, but makes
    a trivial update to the initial parameters given, and
    keeps record of the number of iterations made.
    '''

    def __init__(self, w_init, t_max):
        
        self.w = w_init
        self.t_max = t_max


    def __iter__(self):

        self.t = 0
        print("(__iter__): t =", self.t)

        return self
    

    def __next__(self):

        # Condition for stopping.
        
        if self.t >= self.t_max:
            print("--- Condition reached! ---")
            raise StopIteration

        print("(__next__): t =", self.t)
        self.w *= 2
        self.t += 1

        # Note that __next__ does not need to return anything.

    def __str__(self):

        out = "State of w:" + "\n" + "  " + str(self.w)
        return out



class Algo_GD_FiniteDiff:

    '''
    Iterator which implements a line-search steepest descent method,
    via finite differences to approximate the gradient.
    '''

    def __init__(self, w_init, t_max, step, delta, verbose, store):

        # Store the user-supplied information.
        self.w = w_init
        self.t = None
        self.t_max = t_max
        self.step = step
        self.delmtx = np.eye(self.w.size) * delta
        self.delta = delta
        self.verbose = verbose
        self.store = store
        
        # If asked to store, keep record of all updates.
        if self.store:
            self.wstore = np.zeros((self.w.size,t_max+1), dtype=np.float64)
            self.wstore[:,0] = self.w.flatten()
        else:
            self.wstore = None
        

    def __iter__(self):

        self.t = 0

        if self.verbose:
            print("(via __next__)")
            self.print_state()
        
        return self

    
    def __next__(self):

        # Condition for stopping.
        if self.t >= self.t_max:
            if self.verbose:
                print("--- Condition reached! ---")
            raise StopIteration

        self.t += 1

        if self.verbose:
            print("(via __next__)")
            self.print_state()


    def update(self, model):
        
        stepsize = self.step(self.t)
        newdir = np.zeros(self.w.size, dtype=self.w.dtype)
        w_delta = np.copy(self.w)
        w_plus = np.copy(self.w)
        w_minus = np.copy(self.w)
        loss = model.l_tr(self.w)

        # Perturb one coordinate at a time, compute finite difference.
        for j in range(self.w.size):
            
            # Perturb one coordinate. MODEL ACCESS here.
            delj = np.take(self.delmtx,[j],axis=1)
            loss_delta = model.l_tr((self.w + delj))
            
            newdir[j] = np.mean(loss_delta-loss) / self.delta
            
            
        self.w = self.w - stepsize * newdir.reshape(self.w.shape)
        
        if self.store:
            self.wstore[:,self.t] = self.w.flatten()


    def print_state(self):
        print("------------")
        print("t =", self.t, "( max = ", self.t_max, ")")
        print("w = ", self.w)
        print("------------")


def alpha_fixed(t, val):
    '''
    Step-size function: constant.
    '''
    return val

def alpha_log(t, val=1):
    '''
    Step-size function: logarithmic.
    '''
    return val / (1+math.log((1+t)))

def alpha_pow(t, val=1, pow=0.5):
    '''
    Step-size function: polynomial.
    '''
    return val / (1 + t**pow)
