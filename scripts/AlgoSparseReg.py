
import numpy as np

def soft_thres(u,mar):
    '''
    The so-called "soft threshold" function, as made
    popular by the LASSO model and all related
    learning procedures.

    Input "u" will be an array, and "mar" will be the
    margin of the soft-threshold, a non-negative real
    value.
    '''
    
    return np.sign(u) * np.clip(a=(np.abs(u)-mar), a_min=0, a_max=None)




class Algo_LASSO_CD:

    '''
    Coordinate descent (CD) implementation for minimization
    of the "LASSO" objective, namely the sum of squared errors
    regularized by an l1 penalty.
    '''

    def __init__(self, w_init, t_max, lam_l1, verbose):

        # Store the user-supplied information.
        self.w = np.copy(w_init)
        self.t = 0
        self.t_max = t_max
        self.idxsize = self.w.size
        self.lam_l1 = lam_l1
        self.verbose = verbose
    
    
    def __iter__(self):

        self.t = 0
        
        # Shuffle up the indices before starting.
        self.idx = np.random.choice(self.w.size, size=self.w.size, replace=False)

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

        idx_j = self.idx[((self.t-1) % self.idxsize)] # circuits around shuffled coords.
    
        self.w[idx_j] = 0 # current para, but with jth coord set to zero.
        
        g_j = -np.mean(model.g_j_tr(j=idx_j, w=self.w, lam_l1=0))

        # Compute the solution to the one-dimensional optimization,
        # using it to update the parameters.
        self.w[idx_j] = soft_thres(u=g_j, mar=self.lam_l1)
        # --- DB interlude start --- #
        #print("g_j =", g_j, "/ newval =", newval, "lam_l1 =", self.lam_l1)
        # --- DB interlude end --- #
        
        # NOTE: since taking the SUM of losses rather than the mean,
        # there is no factor of n on the "margin" value.
        
        
    def print_state(self):
        print("------------")
        print("t =", self.t, "( max = ", self.t_max, ")")
        print("Index is t =", ((self.t-1) % self.w.size), "of", self.w.size)
        print("w = ", self.w)
        print("------------")
