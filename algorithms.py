
import numpy as np
import math

# Algorithms from Algo_FirstOrder notebook:

class Algo_LineSearch:
    '''
    Basic archetype of an iterator for implementing a line
    search algorithm.

    Note that we assume w_init is a nparray matrix with the
    shape (d,1), where d represents the number of total
    parameters to be determined.
    '''

    def __init__(self, w_init, step, t_max, thres,
                 verbose=False, store=False, lamreg=None):

        # Attributes passed from user.
        self.w = w_init
        self.step = step
        self.t_max = t_max
        self.thres = thres
        self.verbose = verbose
        self.store = store
        self.lamreg = lamreg

        # Attributes determined internally.
        if self.w is None:
            self.nparas = None
            self.w_old = None
        else:
            self.nparas = self.w.size
            self.w_old = np.copy(self.w)
        self.t = None
        self.diff = np.inf
        self.stepcost = 0 # for per-step costs.
        self.cumcost = 0 # for cumulative costs.

        # Keep record of all updates (optional).
        if self.store:
            self.wstore = np.zeros((self.w.size,t_max+1), dtype=self.w.dtype)
            self.wstore[:,0] = self.w.flatten()
        else:
            self.wstore = None
        
        
    def __iter__(self):

        self.t = 0

        if self.verbose:
            print("(via __iter__)")
            self.print_state()
            
        return self
    

    def __next__(self):
        '''
        Check the stopping condition(s).
        '''
        
        if self.t >= self.t_max:
            raise StopIteration

        if self.diff <= self.thres:
            raise StopIteration

        if self.verbose:
            print("(via __next__)")
            self.print_state()
            
            
    def update(self, model, data):
        '''
        Carry out the main parameter update.
        '''
        
        # Parameter update.
        newdir = self.newdir(model=model, data=data)
        stepsize = self.step(t=self.t, model=model, data=data, newdir=newdir)
        self.w = self.w + stepsize * np.transpose(newdir)

        # Update the monitor attributes.
        self.monitor_update(model=model, data=data)

        # Run cost updates.
        self.cost_update(model=model, data=data)

        # Keep record of all updates (optional).
        if self.store:
            self.wstore[:,self.t] = self.w.flatten()


    def newdir(self, model, data):
        '''
        This will be implemented by sub-classes
        that inherit this class.
        '''
        raise NotImplementedError
    

    def monitor_update(self, model, data):
        '''
        This will be implemented by sub-classes
        that inherit this class.
        '''
        raise NotImplementedError

    
    def cost_update(self, model, data):
        '''
        This will be implemented by sub-classes
        that inherit this class.
        '''
        raise NotImplementedError

    
    def print_state(self):
        print("t =", self.t, "( max =", self.t_max, ")")
        print("diff =", self.diff, "( thres =", self.thres, ")")
        if self.verbose:
            print("w = ", self.w)
        print("------------")


class Algo_GD(Algo_LineSearch):
    '''
    Iterator which implements a line-search steepest descent method,
    where the direction of steepest descent is measured using the
    Euclidean norm (this is the "usual" gradient descent). Here the
    gradient is a sample mean estimate of the true risk gradient.
    That is, this is ERM-GD.
    '''

    def __init__(self, w_init, step, t_max, thres, store, lamreg):

        super(Algo_GD, self).__init__(w_init=w_init,
                                      step=step,
                                      t_max=t_max,
                                      thres=thres,
                                      store=store,
                                      lamreg=lamreg)

    def newdir(self, model, data):
        '''
        Determine the direction of the update.
        '''
        return (-1) * np.mean(model.g_tr(w=self.w,
                                         data=data,
                                         lamreg=self.lamreg),
                              axis=0, keepdims=True)

    def monitor_update(self, model, data):
        '''
        Update the counters and convergence
        monitors used by the algorithm. This is
        executed once every step.

        For GD, increment counter and check the
        differences at each step.
        '''
        self.t += 1
        self.diff = np.linalg.norm((self.w-self.w_old))
        self.w_old = np.copy(self.w)

        
    def cost_update(self, model, data):
        '''
        Update the amount of computational resources
        used by the routine.

        Cost here is number of gradient vectors
        computed. GD computes one vector for each
        element in the sample.
        '''
        self.stepcost = data.n_tr
        self.cumcost += self.stepcost



class Algo_FDGD(Algo_LineSearch):
    '''
    Finite-difference gradient descent.
    '''

    def __init__(self, w_init, step, delta, t_max, thres, store, lamreg):

        super(Algo_FDGD, self).__init__(w_init=w_init,
                                        step=step,
                                        t_max=t_max,
                                        thres=thres,
                                        store=store,
                                        lamreg=lamreg)
        self.delta = delta
        self.delmtx = np.eye(self.w.size) * delta
        
    
    def newdir(self, model, data):
        '''
        Determine the direction of the update.
        '''
        out = np.zeros((1,self.w.size), dtype=self.w.dtype)
        loss = model.l_tr(w=self.w, data=data)
        
        # Perturb one coordinate at a time, then 
        # compute finite difference.
        for j in range(self.w.size):
            delj = np.take(self.delmtx,[j],axis=1)
            loss_delta = model.l_tr((self.w + delj),
                                    data=data,
                                    lamreg=self.lamreg)
            out[:,j] = np.mean(loss_delta-loss) / self.delta
            
        return out * (-1)
    

    def monitor_update(self, model, data):
        self.t += 1
        self.diff = np.linalg.norm((self.w-self.w_old))
        self.w_old = np.copy(self.w)
        
        
    def cost_update(self, model, data):
        self.stepcost = data.n_tr
        self.cumcost += self.stepcost



class Algo_SGD(Algo_LineSearch):
    '''
    Stochastic companion to Algo_GD, where we randomly take
    sub-samples (called mini-batches) to compute the sample
    mean estimate of the underlying risk gradient. By using
    small mini-batches, the computational burden of each
    iteration is eased, at the cost of poorer statistical
    estimates.
    '''
    def __init__(self, w_init, step, batchsize, replace,
                 t_max, thres, store, lamreg):

        super(Algo_SGD, self).__init__(w_init=w_init,
                                       step=step,
                                       t_max=t_max,
                                       thres=thres,
                                       store=store,
                                       lamreg=lamreg)
        self.batchsize = batchsize
        self.replace = replace

        # Computed internally.
        self.nseen = 0
        self.npasses = 0
        self.torecord = True


    def newdir(self, model, data):
        '''
        Determine the direction of the update.
        '''
        shufidx = np.random.choice(data.n_tr,
                                   size=self.batchsize,
                                   replace=self.replace)
        return (-1) * np.mean(model.g_tr(w=self.w,
                                         data=data,
                                         n_idx=shufidx,
                                         lamreg=self.lamreg),
                              axis=0, keepdims=True)
    
    def monitor_update(self, model, data):
        '''
        Update the counters and convergence
        monitors used by the algorithm. This is
        executed once every step.
        '''
        self.t += 1
        self.nseen += self.batchsize
        if self.nseen >= data.n_tr:
            self.torecord = True
            self.npasses += 1
            self.diff = np.linalg.norm((self.w-self.w_old))
            self.w_old = np.copy(self.w)
            self.nseen = self.nseen % data.n_tr
    
    def cost_update(self, model, data):
        '''
        Update the amount of computational resources
        used by the routine.

        Cost here is number of gradient vectors
        computed. SGD computes one for each element
        of the mini-batch used.
        '''
        self.stepcost = self.batchsize
        self.cumcost += self.stepcost


class Algo_SVRG(Algo_LineSearch):
    '''
    Stochastic variance reduced gradient descent.
    '''
    def __init__(self, w_init, step, batchsize, replace,
                 out_max, in_max, thres, store, lamreg):

        super(Algo_SVRG, self).__init__(w_init=w_init,
                                        step=step,
                                        t_max=(out_max*in_max),
                                        thres=thres,
                                        store=store,
                                        lamreg=lamreg)
        self.out_max = out_max
        self.in_max = in_max
        self.batchsize = batchsize
        self.replace = replace

        # Computed internally.
        self.nseen = 0
        self.npasses = 0
        self.idx_inner = 0
        self.torecord = True


    def newdir(self, model, data):
        '''
        Determine the direction of the update.
        '''

        if self.idx_inner == 0:
            self.w_snap = np.copy(self.w)
            self.g_snap = np.mean(model.g_tr(w=self.w_snap,
                                             data=data,
                                             lamreg=self.lamreg),
                                  axis=0, keepdims=True)
        
        shufidx = np.random.choice(data.n_tr,
                                   size=self.batchsize,
                                   replace=self.replace)
        g_sgd = np.mean(model.g_tr(w=self.w,
                                   n_idx=shufidx,
                                   data=data,
                                   lamreg=self.lamreg),
                        axis=0, keepdims=True)
        correction = np.mean(model.g_tr(w=self.w_snap,
                                        n_idx=shufidx,
                                        data=data,
                                        lamreg=self.lamreg),
                             axis=0, keepdims=True) - self.g_snap
        return (-1) * (g_sgd-correction)


    def monitor_update(self, model, data):
        '''
        Update the counters and convergence
        monitors used by the algorithm. This is
        executed once every step.
        '''
        self.t += 1
        self.idx_inner += 1
        if self.idx_inner == self.in_max:
            self.idx_inner = 0

        # Check differences every "epoch" over data.
        self.nseen += self.batchsize
        if self.nseen >= data.n_tr:
            self.torecord = True
            self.npasses += 1
            self.diff = np.linalg.norm((self.w-self.w_old))
            self.w_old = np.copy(self.w)
            self.nseen = self.nseen % data.n_tr
            
    
    def cost_update(self, model, data):
        '''
        Update the amount of computational resources
        used by the routine.

        Cost computation based on number of gradients computed:
        - Each inner loop step requires mini-batch gradients for
          two vectors, w and w_snap.
        - Each outer loop step additionally requires a full
          batch of gradients.
        '''

        if self.idx_inner == self.in_max:
            self.stepcost = 2 * self.batchsize + data.n_tr
        else:
            self.stepcost = 2 * self.batchsize
            
        self.cumcost += self.stepcost




class Algo_CDL1:
    '''
    Coordinate descent (CD) implementation for minimization
    of the "LASSO" objective, namely the sum of squared errors
    regularized by an l1 penalty.
    '''
    
    def __init__(self, w_init, t_max, lamreg):
        self.w = w_init
        self.t = None
        self.t_max = t_max
        self.lamreg = lamreg
        
    def __iter__(self):
        self.t = 0
        # Shuffle up the indices before starting.
        self.idx = np.random.choice(self.w.size, size=self.w.size, replace=False)
        self.idxj = self.idx[0]
        return self
        
    def __next__(self):
        if self.t >= self.t_max:
            raise StopIteration

    def update(self, model, data):
        
        # Computations related to the update.
        n = data.X_tr.shape[0]
        modidx = (self.t-1) % self.w.size
        self.idxj = self.idx[modidx] # circuits around shuffled coords.
        self.w[self.idxj,0] = 0 # current para, but with jth coord set to zero.
        g_j = -np.mean(model.g_j_tr(j=self.idxj, w=self.w, data=data, lamreg=0))
        g_j = g_j * n / (n-1) # rescale
        
        # Compute the solution to the one-dimensional optimization,
        # using it to update the parameters.
        self.w[self.idxj,0] = soft_thres(u=g_j, mar=self.lamreg)
        
        # Monitor update.
        self.t += 1
