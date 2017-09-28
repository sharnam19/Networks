import numpy as np
from scipy.stats import threshold
def relu_forward(x):
    """
    Input : x of any shape
    Output: Of same shape as x after applying Relu Non Linearity
    """
    out = np.maximum(0,x)
    cache = (x)
    return out,cache

def relu_backward(dOut,cache):
    """
    Input: dOut of any shape is the upstream gradient
    Output:dx of same shape as x
    """
    x = cache
    dx = np.sign(np.maximum(0,x))*dOut
    return dx

def sigmoid_forward(x):
    """
    Input x of any shape

    Output: Sigmoided x
    """
    out = 1+np.exp(-x)
    out = 1/out
    cache = (x,out)
    return out,cache

def sigmoid_backward(dOut,cache):
    """
    Input:
    dOut of any shape

    Output:
    dx of same shape as dOut
    """
    _, out = cache
    dx = out*(1-out)*dOut
    return dx

def tanh_forward(x):
    """
    Input:
        x of any shape
    Output:
        tanh(x) of same shape as x
        cache containing x
    """
    out = np.tanh(x)
    cache = (out)
    return out,cache

def tanh_backward(dOut,cache):
    """
    Input:
        dOut of same shape as x
        cache containing x
        
    Output:
        dx of same shape as x
    """
    
    (out) = cache
    dx = (1-out*out)*dOut
    return dx

def leaky_relu_forward(x):
    """
    Input:
        x of any shape
    Output:
        out:max(0.1x,x) of same shape as x
        cache:x,out
    """
    
    out = np.maximum(0.1*x,x)
    cache = (x)
    return out,cache

def leaky_relu_backward(dOut,cache):
    """
    Input:
        dOut of any shape
        cache:(x,out)
        x is the input given for forward
        out is the leaky_relu_forwarded output
    Output:
        dx of same shape as x
    """
    (x) = cache
    x = np.sign(x)
    dx = threshold(x,threshmin=0,newval=0.1)*dOut
    return dx