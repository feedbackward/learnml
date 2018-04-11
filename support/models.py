
import support.classes as classes
import numpy as np
from scipy import stats


class LinReg(classes.Data):

    def __init__(self, dinfo):
        '''
        Given data with X_tr, X_te, y_tr, y_te, we
        initialize a model object with loss functions, gradients,
        Hessians, etc., as well as an "eval" method which
        automatically knows to use the test data.
        '''
        # Given data info, load it up into memory for use.
        super(LinReg,self).__init__(dinfo)
        self.n = self.X_tr.shape[0]
        self.d = self.X_tr.shape[1]


    def __str__(self):
        s_mod = "MODEL: Linear regression."\
                + "\n" + "Info on data as follows..."
        s_data = super(LinReg,self).__str__()
        return s_mod + "\n" + s_data

    
    def w_initialize(self):

        # Randomly generated uniformly on [-1,1].
        out = 2 * np.random.random_sample((self.d,1)) - 1
        return out
    

    def eval(self, w):
        '''
        Evaluate a parameter vector on the test data.
        '''
        # Specify the loss to use here.
        losses = self.l_te(w=w)

        # Specify the loss-based statistics to use here.
        rawres = [losses.mean(), losses.std()]
        
        return rawres


    def l_imp(self, w, X, y, lam_l1=0, lam_l2=0):
        '''
        Implementation of (regularized) squared error as
        loss function under a linear model.
        '''
        
        # Args:
        # w is a (d x 1) matrix taking real values.
        # X is a (k x d) matrix of n observations.
        # y is a (k x 1) matrix taking real values.
        # lam_* are parameters for l1/l2 regularization.

        # Output: (n x 1) losses at each point.

        # Compute regularization terms if required.
        if lam_l1 > 0:
            l1reg = lam_l1 * np.sum(np.abs(w)) # l1 norm
        else:
            l1reg = 0
        if lam_l2 > 0:
            l2reg = lam_l2 * np.sum(w*w) # squared l2
        else:
            l2reg = 0

        return (np.dot(X,w)-y)**2 + l1reg + l2reg

    
    def l_tr(self, w, lam_l1=0, lam_l2=0):
        return self.l_imp(w=w, X=self.X_tr, y=self.y_tr,
                          lam_l1=lam_l1, lam_l2=lam_l2)

    def l_te(self, w, lam_l1=0, lam_l2=0):
        return self.l_imp(w=w, X=self.X_te, y=self.y_te,
                          lam_l1=lam_l1, lam_l2=lam_l2)
    
    
    def g_imp(self, w, X, y, lam_l1=0, lam_l2=0):
        '''
        Gradient of the above loss function.
        '''
        # Args:
        # w is a (d x 1) matrix taking real values.
        # X is a (k x d) matrix of n observations.
        # y is a (k x 1) matrix taking real values.
        # lam_* are parameters for l1/l2 regularization.
        
        # Output: (n x d) matrix of gradient vecs at each point.

        # Compute regularization terms if required.
        # note: transpose is because we need to add to all rows.
        if lam_l1 > 0:
            g_l1reg = lam_l1 * (np.sign(w)).transpose()
        else:
            g_l1reg = 0
        if lam_l2 > 0:
            g_l2reg = 2 * lam_l2 * w.transpose() # grad of squared l2 norm.
        else:
            g_l2reg = 0

        return -(y-np.dot(X,w)) * X + g_l1reg + g_l2reg

    def g_tr(self, w, lam_l1=0, lam_l2=0):
        return self.g_imp(w=w, X=self.X_tr, y=self.y_tr,
                          lam_l1=lam_l1, lam_l2=lam_l2)

    def g_te(self, w, lam_l1=0, lam_l2=0):
        return self.g_imp(w=w, X=self.X_te, y=self.y_te,
                          lam_l1=lam_l1, lam_l2=lam_l2)

    def g_j_imp(self, j, w, X, y, lam_l1=0, lam_l2=0):
        '''
        One coordinate of the gradient of the above
        loss function, evaluated at all data points
        provided.

        This function is efficient when we only need one
        coordinate at a time.
        '''
        # Args:
        # w is a (d x 1) matrix taking real values.
        # X is a (k x d) matrix of n observations.
        # y is a (k x 1) matrix taking real values.
        # lam_* are parameters for l1/l2 regularization.
        
        # Output: (n x 1) matrix of gradient coords at each point.

        # Compute regularization terms if required.
        # note: transpose is because we need to add to all rows.
        if lam_l1 > 0:
            g_l1reg = lam_l1 * np.sign(w[j,0])
        else:
            g_l1reg = 0
        if lam_l2 > 0:
            g_l2reg = 2 * lam_l2 * w[j,0] # grad of squared l2 norm.
        else:
            g_l2reg = 0

        return -(y-np.dot(X,w)) * np.take(X,[j],1) + g_l1reg + g_l2reg
    
    def g_j_tr(self, j, w, lam_l1=0, lam_l2=0):
        return self.g_j_imp(j=j, w=w, X=self.X_tr, y=self.y_tr,
                            lam_l1=lam_l1, lam_l2=lam_l2)

    def g_j_te(self, j, w, lam_l1=0, lam_l2=0):
        return self.g_j_imp(j=j, w=w, X=self.X_te, y=self.y_te,
                            lam_l1=lam_l1, lam_l2=lam_l2)

    def corr_imp(self, w, X, y):
        '''
        Wrapper for Pearson's correlation coefficient,
        computed for the predicted response and the
        actual response.
        '''
        
        # Args:
        # w is a (d x 1) matrix taking real values.
        # X is a (k x d) matrix of n observations.
        # y is a (k x 1) matrix taking real values.

        # Output: a real-valued correlation coefficient.

        yest = np.dot(X,w)
        return stats.pearsonr(yest.flatten(), y.flatten())[0]

    def corr_tr(self, w):
        return self.corr_imp(w=w, X=self.X_tr, y=self.y_tr)

    def corr_te(self, w):
        return self.corr_imp(w=w, X=self.X_te, y=self.y_te)


class Encoder(LinReg):

    def __init__(self, dinfo):
        '''
        This is an application-specific class for the motion
        energy encoder. It inherits the linear regression model,
        and the only difference is that we extract individual
        voxels at the point of initialization.
        '''
        super(Encoder,self).__init__(dinfo)

        # Extract a single voxel's worth of data.
        # NOTE: assumes the shape is (#voxels, #points).
        self.voxidx = dinfo.misc["voxidx"]
        self.y_tr = np.transpose(np.take(self.y_tr, [self.voxidx], 0))
        self.y_te = np.transpose(np.take(self.y_te, [self.voxidx], 0))
    

class NoisyOpt(classes.Data):

    def __init__(self, dinfo):
        '''
        Model object for general-purpose noisy optimization
        demo, where we just have training data and oracle
        information for a risk function.
        '''
        # Given data info, load it up into memory for use.
        super(NoisyOpt,self).__init__(dinfo)
        self.n = self.X_tr.shape[0]
        self.d = self.X_tr.shape[1]
        self.nsub = dinfo.misc["nsub"]

        # Given oracle information, use it for later evaluation.
        self.sigma_noise = dinfo.misc["sigma_noise"]
        self.cov_X = dinfo.misc["cov_X"]
        self.w_true = dinfo.misc["w_true"]
        self.w_init = dinfo.misc["w_init"]


    def __str__(self):
        s_mod = "MODEL: NoisyOpt."\
                + "\n" + "Info on data as follows..."
        s_data = super(NoisyOpt,self).__str__()
        return s_mod + "\n" + s_data

    
    def w_initialize(self):

        return self.w_init # initial vals set at time of data gen.


    def eval(self, w):
        '''
        Evaluate a parameter vector using oracle information
        on the true underlying objective. The form below assumes
        only that both the inputs and additive noise have zero
        mean, and that the inputs are independent of the noise.
        There is no restriction on correlation between the
        components of the input.
        '''
        diff = w - self.w_true
        quad = np.dot(diff.transpose(),\
                      np.dot(self.cov_X, diff).reshape(diff.shape))

        return np.float64(quad) + self.sigma_noise

    def eval2D_helper(self, w1, w2):
        w2D = np.array([w1,w2]).reshape((2,1))
        return self.eval(w=w2D)

    def evalDist(self, w):
        '''
        The l2 norm of the difference between the estimate
        vector and the true underlying model vector.
        '''
        return np.linalg.norm(w-self.w_true)

    def evalSparsity(self, w):
        '''
        The l0 norm of the estimate vector.
        '''
        return np.count_nonzero(w)


    def l_imp(self, w, X, y, lam_l1=0, lam_l2=0):
        '''
        Implementation of (regularized) squared error as
        loss function under a linear model.
        '''
        # Args:
        # w is a (d x 1) matrix taking real values.
        # X is a (k x d) matrix of n observations.
        # y is a (k x 1) matrix taking real values.
        # lam_* are parameters for l1/l2 regularization.

        # Output: (n x 1) losses at each point.

        # Compute regularization terms if required.
        if lam_l1 > 0:
            l1reg = lam_l1 * np.sum(np.abs(w)) # l1 norm
        else:
            l1reg = 0
        if lam_l2 > 0:
            l2reg = lam_l2 * np.sum(w*w) # squared l2
        else:
            l2reg = 0

        return (np.dot(X,w)-y)**2 / 2 + l1reg + l2reg


    def l_tr(self, w, lam_l1=0, lam_l2=0):
        return self.l_imp(w=w, X=self.X_tr, y=self.y_tr,
                          lam_l1=lam_l1, lam_l2=lam_l2)


    def g_imp(self, w, X, y, lam_l1=0, lam_l2=0):
        '''
        Gradient of the above loss function.
        '''
        # Args:
        # w is a (d x 1) matrix taking real values.
        # X is a (k x d) matrix of n observations.
        # y is a (k x 1) matrix taking real values.
        # lam_* are parameters for l1/l2 regularization.
        
        # Output: (n x d) matrix of gradient vecs at each point.

        # Compute regularization terms if required.
        # note: transpose is because we need to add to all rows.
        if lam_l1 > 0:
            g_l1reg = lam_l1 * (np.sign(w)).transpose()
        else:
            g_l1reg = 0
        if lam_l2 > 0:
            g_l2reg = 2 * lam_l2 * w.transpose() # grad of squared l2 norm.
        else:
            g_l2reg = 0

        return -(y-np.dot(X,w)) * X + g_l1reg + g_l2reg
    

    def g_tr(self, w, lam_l1=0, lam_l2=0):
        return self.g_imp(w=w, X=self.X_tr, y=self.y_tr,
                          lam_l1=lam_l1, lam_l2=lam_l2)

    
    def g_j_imp(self, j, w, X, y, lam_l1=0, lam_l2=0):
        '''
        One coordinate of the gradient of the above
        loss function, evaluated at all data points
        provided.

        This function is efficient when we only need one
        coordinate at a time.
        '''
        # Args:
        # w is a (d x 1) matrix taking real values.
        # X is a (k x d) matrix of n observations.
        # y is a (k x 1) matrix taking real values.
        # lam_* are parameters for l1/l2 regularization.
        
        # Output: (n x 1) matrix of gradient coords at each point.

        # Compute regularization terms if required.
        # note: transpose is because we need to add to all rows.
        if lam_l1 > 0:
            g_l1reg = lam_l1 * np.sign(w[j,0])
        else:
            g_l1reg = 0
        if lam_l2 > 0:
            g_l2reg = 2 * lam_l2 * w[j,0] # grad of squared l2 norm.
        else:
            g_l2reg = 0

        return -(y-np.dot(X,w)) * np.take(X,[j],1) + g_l1reg + g_l2reg
    
    def g_j_tr(self, j, w, lam_l1=0, lam_l2=0):
        return self.g_j_imp(j=j, w=w, X=self.X_tr, y=self.y_tr,
                            lam_l1=lam_l1, lam_l2=lam_l2)

