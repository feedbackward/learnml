
import support.classes as classes
import numpy as np
import scipy
from scipy import stats

class LgstReg(classes.Data):

    def __init__(self, dinfo):
        '''
        Given data with X_tr, X_te, y_tr, y_te, we
        initialize a model object with loss functions, gradients,
        Hessians, etc., as well as an "eval" method which
        automatically knows to use the test data.
        '''
        # Given data info, load up the (X,y) data.
        super(LgstReg,self).__init__(dinfo)

        # Convert original labels to a one-hot binary representation.
        self.nc = self.get_nc() # get the number of classes.
        self.C_tr = self.onehot(y=self.y_tr) # one-hot training labels.
        self.C_te = self.onehot(y=self.y_te) # one-hot testing labels.
        self.n = self.X_tr.shape[0] # number of training obs.
        self.d_feat = self.X_tr.shape[1] # number of features.
        self.d_para = self.d_feat * (self.nc-1) # number of parameters to set.
        
        
    def __str__(self):
        s_mod = "MODEL: Logistic regression."\
                + "\n" + "Info on data as follows..."
        s_data = super(LgstReg,self).__str__()
        return s_mod + "\n" + s_data

    
    def w_initialize(self):

        # Randomly generated (uniformly on [-1,1].
        out = 2 * np.random.random_sample( (self.d_para,1) ) - 1
        return out

    
    def get_nc(self):
        '''
        Get the number of classes.
        '''
        return np.unique(np.concatenate( (self.y_tr, self.y_te), axis=0)).size

    
    def onehot(self, y):
        '''
        A function for encoding y into a one-hot vector.
        '''

        # Inputs:
        # y is a (k x 1) array, taking values in {0,1,...,nc-1}.
        # NOTE: we say "k" here because it may be the training,
        #       test, or both training/test labels together.

        nc = self.nc
        k = y.size
        C = np.zeros(nc*k, dtype=np.int16).reshape( (k,nc) )

        for i in range(k):
            print("y:", y)
            j = y[i,0] # assumes y has only one column.
            C[i,j] = 1

        return C

    
    def eval(self, w):
        '''
        Evaluate a parameter vector on the test data.
        '''

        losses = self.l_te(w=w) # logistic reg loss.

        # Based on pre-specified decision rule, get classification rate.
        y_est = self.classify(w=w, X=self.X_te)
        perf = self.class_perf(y_est, self.y_te)
        
        # Specify the loss-based statistics to use here.
        rawres = [losses.mean(), losses.std(), perf["rate"]]
        # potential extension: can add per-class P/R/F1 if desired.
        
        return rawres
    

    def class_perf(self, y_est, y_true):
        '''
        Given class label estimates and true values, compute the
        fraction of correct classifications made.
        '''
        
        # Input:
        # y_est and y_true are (k x 1) matrices of labels.

        # Output:
        # Returns a dictionary with two components, (1) being
        # the fraction of correctly classified labels, and
        # (2) being a dict of per-label precison/recall/F1
        # scores. 

        # First, get the classification rate.
        k = y_est.size
        num_correct = (y_est == y_true).sum()
        frac_correct = num_correct / k

        # Then, get precision/recall for each class.
        prec_rec = { i:None for i in range(self.nc) } # initialize

        for c in range(self.nc):

            idx_c = (y_true == c)
            idx_notc = (idx_c == False)

            TP = (y_est[idx_c] == c).sum()
            FN = idx_c.sum() - TP
            FP = (y_est[idx_notc] == c).sum()
            TN = idx_notc.sum() - FP

            # Precision.
            if (TP == 0 and FP == 0):
                prec = 0
            else:
                prec = TP / (TP+FP)

            # Recall.
            if (TP == 0 and FN == 0):
                rec = 0
            else:
                rec = TP / (TP+FN)

            # F1 (harmonic mean of precision and recall).
            if (prec == 0 or rec == 0):
                f1 = 0
            else:
                f1 = 2 * prec * rec / (prec + rec)

            prec_rec[c] = {"P": prec,
                           "R": rec,
                           "F1": f1}

        return {"rate": frac_correct,
                "PRF1": prec_rec}


    def l_imp(self, w, X, C, lam=0):
        '''
        Implementation of the multi-class logistic regression
        loss function.
        '''
        
        # Input:
        # w is a (d_para x 1) matrix of weights.
        # X is a (k x d_feat) matrix of k observations.
        # C is a (k x nc) matrix giving a binarized encoding of the
        # class labels for each observation; each row a one-hot vector.
        # lam is a non-negative regularization parameter.
        # NOTE: k can be anything, the training/test sample size.

        # Output:
        # A vector of length k with losses evaluated at k points.

        k = X.shape[0]

        # Initialize and populate the activations.
        A = np.zeros(k*self.nc).reshape( (self.nc, k) )
        A[:-1,:] = np.dot(w.reshape((self.nc-1,self.d_feat)), # reshape w.
                          np.transpose(X)) # leave last row as zeros.

        # Raw activations of all the correct weights.
        cvec = np.sum(A*np.transpose(C), axis=0)
        
        # Compute the negative log-likelihoods.
        err = np.log(np.sum(np.exp(A), axis=0)) - cvec

        # Return the losses (all data points), with penalty if needed.
        if (lam > 0):
            return err + lam * np.linalg.norm(W)**2
        else:
            return err
        
    def l_tr(self, w, lam=0):
        return self.l_imp(w=w, X=self.X_tr, C=self.C_tr, lam=lam)

    def l_te(self, w, lam=0):
        return self.l_imp(w=w, X=self.X_te, C=self.C_te, lam=lam)
    
    
    def g_imp(self, w, X, C, lam=0):
        '''
        Implementation of the gradient of the loss function used in
        multi-class logistic regression.
        '''
        
        # Input:
        # w is a (d_para x 1) matrix of weights.
        # X is a (k x d_feat) matrix of k observations.
        # C is a (k x nc) matrix giving a binarized encoding of the
        # class labels for each observation; each row a one-hot vector.
        # lam is a non-negative regularization parameter.
        # NOTE: k can be anything, the training/test sample size.

        # Output:
        # A (k x d_para) matrix of gradients eval'd at k points.

        # Initialize and populate the activations.
        k = X.shape[0]
        
        A = np.zeros(k*self.nc).reshape( (self.nc,k) )
        A[:-1,:] = np.dot(w.reshape((self.nc-1,self.d_feat)), # reshape w.
                          np.transpose(X)) # leave last row as zeros.

        # Compute the conditional label probabilities.
        P = np.exp(A) / np.sum(np.exp(A), axis=0) # (nc x k)
        
        # Initialize a large matrix (k x d_para) to house per-point grads.
        G = np.arange(k*self.d_para).reshape( (k,self.d_para) )

        for i in range(k):
            # A very tall vector (i.e., just one "axis").
            G[i,:] = np.kron(a=(P[:-1,i]-C[i,:-1]), b=X[i,:])
            # NOTE: carefully removing the last elements.

        return G
        
    def g_tr(self, w, lam=0):
        return self.g_imp(w=w, X=self.X_tr, C=self.C_tr, lam=lam)

    def g_te(self, w, lam=0):
        return self.g_imp(w=w, X=self.X_te, C=self.C_te, lam=lam)

    
    def h_imp(self, w, lam=0):
        pass
    
    def h_tr(self, w, lam=0):
        pass

    def h_te(self, w, lam=0):
        pass

    def classify(self, w, X):
        '''
        Given learned weights (w) and a matrix of one or
        more observations, classify them as {0,...,nc-1}.
        '''

        # Input:
        # w is a (d_para x 1) matrix of weights.
        # X is a (k x d_feat) matrix of k observations.
        # NOTE: k can be anything, the training/test sample size.

        # Output:
        # A vector of length k, housing labels in {0,...,nc-1}.

        # Get the activations, and then the conditional probabilities.
        k = X.shape[0]
        A = np.zeros(k*self.nc).reshape( (self.nc,k) )
        A[:-1,:] = np.dot(w.reshape((self.nc-1,self.d_feat)), # reshape w.
                          np.transpose(X)) # leave last row as zeros.
        P = np.exp(A) / np.sum(np.exp(A), axis=0) # (nc x k)

        # Return the class with the largest prob, given the data.
        return np.argmax(P, axis=0).reshape( (k,1) )


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
        return scipy.stats.pearsonr(yest.flatten(), y.flatten())[0]

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

