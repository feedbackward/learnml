
import numpy as np
import chainer as ch


# FunctionNode sub-class, for implementing a fully connected linear layer.

class LinearFunction(ch.function_node.FunctionNode):
    '''
    FunctionNode object, defined on Variable objects,
    which is the basis for a linear transformation to
    be wrapped up as a Link object.
    
    Take d-dimensional inputs x, and using array W of
    shape (k,d), map this input to k outputs. That is,
    we have a fully-connected layer with d input units
    and k output units.
    '''
    
    def forward(self, inputs):
        '''
        Forward computation for both CPU and GPU.
        '''
        
        # Unpack the tuple of inputs.
        if len(inputs) == 3:
            x, W, b = inputs
        else:
            (x, W), b = inputs, None

        y = x.dot(W.T).astype(x.dtype, copy=False)
        
        # Add a bias term, if relevant.
        if b is not None:
            y += b
            
        # Since backward() depends only on x and W,
        # we need only retain these two.
        self.retain_inputs((0,1))
        
        # Must return the output as a tuple.
        return (y,)

    def backward(self, indices, grad_outputs):
        '''
        General-purpose computation for both CPU/GPU.
        '''
        
        x, W = self.get_retained_inputs()
        gy, = grad_outputs # written as gamma in their docs.
        
        # Docs say that backward() must return a tuple, but
        # looking at their source code for linear.py, it
        # seems like lists are fine.
        out = []
        if 0 in indices:
            gx = gy @ W
            out.append(ch.functions.cast(gx, x.dtype))
        if 1 in indices:
            gW = gy.T @ x
            out.append(ch.functions.cast(gW, W.dtype))
        if 2 in indices:
            # Summing here is simple: for n observations,
            # gy has shape (n,k), where k is the number of
            # layer outputs. Summing over axis=0 is summing
            # over OBSERVATIONS, not over outputs.
            gb = ch.functions.sum(gy, axis=0)
            
        # Return just the relevant gradients we appended.
        return out


def linear(x, W, b):
    '''
    A nice thin wrapper for our linear FunctionNode on
    Variable objects.
    '''
    
    if b is None:
        args = (x, W)
    else:
        args = (x, W, b)
    
    # Don't forget to unpack from the tuple.
    y, = LinearFunction().apply(args)
    return y


# Link object for our linear FunctionNode.

class Linear(ch.Link):
    '''
    A Link class for our linear transformation, implemented
    in the LinearFunction class.
    '''
    def __init__(self,
                 in_size, out_size,
                 init_W=None, init_b=None,
                 init_delta=None,
                 nobias=False):
        super(Linear, self).__init__()
        
        # Here we initialize and "register" the parameters
        # of interest. This is critical because when we
        # call __call__(x) and apply the underlying affine
        # transformations to input x (both forward pass and
        # backward pass), the optimization algorithms knows
        # that we want to optimize W and maybe b, but not x.

        with self.init_scope():
            
            # If provided an ndarray, use it.
            if init_W is not None:
                self.W = ch.Parameter(initializer=np.copy(init_W))
            
            # Else, use a built-in initializer.
            else:
                W_initializer = ch.initializers.Uniform(scale=init_delta,
                                                        dtype=np.float32)
                self.W = ch.Parameter(initializer=W_initializer,
                                      shape=(out_size, in_size))
            
            if nobias:
                self.b = None
            else:
                if init_b is not None:
                    self.b = ch.Parameter(initializer=np.copy(init_b))
                else:
                    self.b = ch.Parameter(initializer=0,
                                          shape=(out_size,))
                
    def __call__(self, x):
        '''
        This method actually applies the linear layer to
        inputs x.
        '''
        return linear(x, self.W, self.b)


# Chain object, composed of Link objects. This is a proper model in the
# sense that it can be fed to optimizers.

class MyChain(ch.Chain):
    '''
    Perhaps the simplest possible model, a
    feed-forward neural network without any
    hidden layers. Just one, fully-connected
    linear layer, aka a classical linear model.
    
    out_l0: number of outputs from layer 0.
    out_l1: number of outputs from layer 1.
    '''
    
    def __init__(self,
                 out_l0,
                 out_l1,
                 init_W=None, init_b=None,
                 init_delta=1.0,
                 nobias=False):
        super(MyChain, self).__init__()
        
        with self.init_scope():
            self.l1 = Linear(in_size=out_l0,
                             out_size=out_l1,
                             init_W=init_W,
                             init_b=init_b,
                             init_delta=init_delta,
                             nobias=True)

    def __call__(self, x):
        return self.l1(x) # parameters are managed by Links.
