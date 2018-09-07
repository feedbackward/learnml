
import numpy as np


class Model:
    '''
    Base class for model objects.
    '''

    def __init__(self, name=None):
        self.name = name

    def l_imp(self, w=None, X=None, y=None, lamreg=None):
        raise NotImplementedError
    
    def l_tr(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.l_imp(w=w, X=data.X_tr,
                              y=data.y_tr,
                              lamreg=lamreg)
        else:
            return self.l_imp(w=w, X=data.X_tr[n_idx,:],
                              y=data.y_tr[n_idx,:],
                              lamreg=lamreg)
    
    def l_te(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.l_imp(w=w, X=data.X_te,
                              y=data.y_te,
                              lamreg=lamreg)
        else:
            return self.l_imp(w=w, X=data.X_te[n_idx,:],
                              y=data.y_te[n_idx,:],
                              lamreg=lamreg)

    def g_imp(self, w=None, X=None, y=None, lamreg=None):
        raise NotImplementedError
    
    def g_tr(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.g_imp(w=w, X=data.X_tr,
                              y=data.y_tr,
                              lamreg=lamreg)
        else:
            return self.g_imp(w=w, X=data.X_tr[n_idx,:],
                              y=data.y_tr[n_idx,:],
                              lamreg=lamreg)
    
    def g_te(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.g_imp(w=w, X=data.X_te,
                              y=data.y_te,
                              lamreg=lamreg)
        else:
            return self.g_imp(w=w, X=data.X_te[n_idx,:],
                              y=data.y_te[n_idx,:],
                              lamreg=lamreg)


class LinReg(Model):
    '''
    General-purpose linear regression model.
    No losses are implemented, just a predict()
    method.
    '''
    
    def __init__(self, data=None, name=None):
        super(LinReg, self).__init__(name=name)
        pass

    def predict(self, w, X):
        '''
        Predict real-valued response.
        w is a (d x 1) array of weights.
        X is a (k x d) matrix of k observations.
        Returns array of shape (k x 1) of predicted values.
        '''
        return X.dot(w)


class LinearL2(LinReg):
    '''
    Orthodox linear regression model, using squared
    error and regularization via the l2 norm.
    '''
    
    def __init__(self, data=None):
        super(LinearL2,self).__init__(data=data)

        
    def l_imp(self, w, X, y, lamreg=None):
        '''
        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeat) matrix of k observations.
        y is a (k x 1) matrix of labels in {-1,1}.
        lamreg is a regularization parameter (l2 penalty).

        Output:
        A vector of length k with losses evaluated at k points.
        '''
        if lamreg is None:
            return (y-self.predict(w=w,X=X))**2/2
        else:
            penalty = lamreg * np.linalg.norm(w)**2
            return (y-self.predict(w=w,X=X))**2/2 + penalty
    
    
    def g_imp(self, w, X, y, lamreg=None):
        '''
        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeat) matrix of k observations.
        y is a (k x 1) matrix of labels in {-1,1}.

        Output:
        A (k x numfeat) matrix of gradients evaluated
        at k points.
        '''
        if lamreg is None:
            return (y-self.predict(w=w,X=X))*(-1)*X
        else:
            penalty = lamreg*2*w.T
            return (y-self.predict(w=w,X=X))*(-1)*X + penalty

class LinearL1(LinReg):
    '''
    Orthodox linear regression model, using squared
    error and regularization via the l1 norm. Good for
    realizing sparsity without giving up convexity.
    '''
    
    def __init__(self, data=None):
        super(LinearL1,self).__init__(data=data)

        
    def l_imp(self, w, X, y, lamreg=None):
        '''
        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeat) matrix of k observations.
        y is a (k x 1) matrix of labels in {-1,1}.
        lamreg is a regularization parameter (l2 penalty).

        Output:
        A vector of length k with losses evaluated at k points.
        '''
        if lamreg is None:
            return (y-self.predict(w=w,X=X))**2/2
        else:
            penalty = lamreg * np.abs(w).sum()
            return (y-self.predict(w=w,X=X))**2/2 + penalty

        
    def g_j_imp(self, j, w, X, y, lamreg=None):

        if lamreg is None:
            return (y-self.predict(w=w,X=X))*(-1)*np.take(a=X,
                                                          indices=[j],
                                                          axis=1)
        else:
            penalty = lamreg * np.sign(w[j,0])
            return (y-self.predict(w=w,X=X))*(-1)*np.take(a=X,
                                                          indices=[j],
                                                          axis=1) + penalty
        
    def g_j_tr(self, j, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.g_j_imp(j=j, w=w, X=data.X_tr,
                                y=data.y_tr,
                                lamreg=lamreg)
        else:
            return self.g_j_imp(j=j, w=w, X=data.X_tr[n_idx,:],
                                y=data.y_tr[n_idx,:],
                                lamreg=lamreg)
    
    def g_j_te(self, j, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.g_j_imp(j=j, w=w, X=data.X_te,
                                y=data.y_te,
                                lamreg=lamreg)
        else:
            return self.g_j_imp(j=j, w=w, X=data.X_te[n_idx,:],
                                y=data.y_te[n_idx,:],
                                lamreg=lamreg)



class Classifier(Model):
    '''
    Generic classifier model, an object with methods
    for both training and evaluating classifiers.
    '''

    def __init__(self, data=None, name=None):
        super(Classifier, self).__init__(name=name)

        # If given data, collect information about labels.
        if data is not None:
            self.labels = self.get_labels(data=data) # all unique labels.
            self.nc = self.labels.size # number of unique labels.


    def onehot(self, y):
        '''
        A function for encoding y into a one-hot vector.
        Inputs:
        - y is a (k,1) array, taking values in {0,1,...,nc-1}.
        '''
        nc = self.nc
        k = y.shape[0]
        C = np.zeros((k,nc), dtype=y.dtype)

        for i in range(k):
            j = y[i,0] # assumes y has only one column.
            C[i,j] = 1

        return C
        

    def get_labels(self, data):
        '''
        Get all the (unique) labels that appear in the data.
        '''
        A = (data.y_tr is None)
        B = (data.y_te is None)

        if (A and B):
            raise ValueError("No label data provided!")
        else:
            if A:
                out_labels = np.unique(data.y_te)
            elif B:
                out_labels = np.unique(data.y_tr)
            else:
                out_labels = np.unique(np.concatenate((data.y_tr,
                                                       data.y_te), axis=0))
            count = out_labels.size
            return out_labels.reshape((count,1))


    def classify(self, X):
        '''
        Must be implemented by sub-classes.
        '''
        raise NotImplementedError


    def class_perf(self, y_est, y_true):
        '''
        Given class label estimates and true values,
        compute the fraction of correct classifications
        made for each label, yielding typical binary
        classification performance metrics.

        Input:
        y_est and y_true are (k x 1) matrices of labels.

        Output:
        Returns a dictionary with two components, (1) being
        the fraction of correctly classified labels, and
        (2) being a dict of per-label precison/recall/F1
        scores. 
        '''
        
        # First, get the classification rate.
        k = y_est.size
        num_correct = (y_est == y_true).sum()
        frac_correct = num_correct / k
        frac_incorrect = 1.0 - frac_correct

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

        return {"rate": frac_incorrect,
                "PRF1": prec_rec}


class LogisticReg(Classifier):
    '''
    Multi-class logistic regression model.
    '''

    def __init__(self, data=None):
        
        # Given data info, load up the (X,y) data.
        super(LogisticReg, self).__init__(data=data)
        
        # Convert original labels to a one-hot binary representation.
        if data.y_tr is not None:
            self.C_tr = self.onehot(y=data.y_tr)
        if data.y_te is not None:
            self.C_te = self.onehot(y=data.y_te)
        
        
    def classify(self, w, X):
        '''
        Given learned weights (w) and a matrix of one or
        more observations, classify them as {0,...,nc-1}.

        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeat) matrix of k observations.
        NOTE: k can be anything, the training/test sample size.

        Output:
        A vector of length k, housing labels in {0,...,nc-1}.
        '''
        
        k, numfeat = X.shape
        A = np.zeros((self.nc,k), dtype=np.float64)
        
        # Get activations, with last row as zeros.
        A[:-1,:] = w.reshape((self.nc-1, numfeat)).dot(X.T)
        
        # Now convert activations to conditional probabilities.
        A = np.exp(A)
        A = A / A.sum(axis=0)  # (nc x k).
        
        # Assign classes with highest probability, (k x 1) array.
        return A.argmax(axis=0).reshape((k,1))


    def l_imp(self, w, X, C, lamreg=None):
        '''
        Implementation of the multi-class logistic regression
        loss function.

        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeat) matrix of k observations.
        C is a (k x nc) matrix giving a binarized encoding of the
        class labels for each observation; each row a one-hot vector.
        lam is a non-negative regularization parameter.
        NOTE: k can be anything, the training/test sample size.

        Output:
        A vector of length k with losses evaluated at k points.
        '''
        
        k, numfeat = X.shape
        A = np.zeros((self.nc,k), dtype=np.float64)
        
        # Get activations, with last row as zeros.
        A[:-1,:] = w.reshape((self.nc-1, numfeat)).dot(X.T)
        
        # Raw activations of all the correct weights.
        cvec = (A*C.T).sum(axis=0)
        
        # Compute the negative log-likelihoods.
        err = np.log(np.exp(A).sum(axis=0))-cvec

        # Return the losses (all data points), with penalty if needed.
        if lamreg is None:
            return err
        else:
            penalty = lamreg * np.linalg.norm(W)**2
            return err + penalty
    
    
    def l_tr(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.l_imp(w=w, X=data.X_tr,
                              C=self.C_tr,
                              lamreg=lamreg)
        else:
            return self.l_imp(w=w, X=data.X_tr[n_idx,:],
                              C=self.C_tr[n_idx,:],
                              lamreg=lamreg)
    
    def l_te(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.l_imp(w=w, X=data.X_te,
                              C=self.C_te,
                              lamreg=lamreg)
        else:
            return self.l_imp(w=w, X=data.X_te[n_idx,:],
                              C=self.C_te[n_idx,:],
                              lamreg=lamreg)
        
        
    def g_imp(self, w, X, C, lamreg=0):
        '''
        Implementation of the gradient of the loss function used in
        multi-class logistic regression.

        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeat) matrix of k observations.
        C is a (k x nc) matrix giving a binarized encoding of the
        class labels for each observation; each row a one-hot vector.
        lamreg is a non-negative regularization parameter.
        NOTE: k can be anything, the training/test sample size.

        Output:
        A (k x d) matrix of gradients eval'd at k points.
        '''
        
        k, numfeat = X.shape
        A = np.zeros((self.nc,k), dtype=np.float64)
        
        # Get activations, with last row as zeros.
        A[:-1,:] = w.reshape((self.nc-1, numfeat)).dot(X.T)
        
        # Now convert activations to conditional probabilities.
        A = np.exp(A)
        A = A / A.sum(axis=0) # (nc x k).
        A = np.float32(A)
        
        # Initialize a large matrix (k x d) to house per-point grads.
        G = np.zeros((k,w.size), dtype=w.dtype)

        for i in range(k):
            # A very tall vector (i.e., just one "axis").
            G[i,:] = np.kron(a=(A[:-1,i]-C[i,:-1]), b=X[i,:])
            # Note we carefully remove the last elements.
        
        if lamreg is None:
            return G
        else:
            return G + lamreg*2*w.T
        

    def g_tr(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.g_imp(w=w, X=data.X_tr,
                              C=self.C_tr,
                              lamreg=lamreg)
        else:
            return self.g_imp(w=w, X=data.X_tr[n_idx,:],
                              C=self.C_tr[n_idx,:],
                              lamreg=lamreg)
    
    def g_te(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.g_imp(w=w, X=data.X_te,
                              C=self.C_te,
                              lamreg=lamreg)
        else:
            return self.g_imp(w=w, X=data.X_te[n_idx,:],
                              C=self.C_te[n_idx,:],
                              lamreg=lamreg)


