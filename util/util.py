import numpy as np
def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def num_gradient_array(f,x,dOut,h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        x[ix]+=h
        fxph=f(x).copy()
        x[ix]-=2*h
        fxmh=f(x).copy()
        x[ix]+=h
        grad[ix] = np.sum((fxph-fxmh)*dOut)/(2*h)
        it.iternext()

    return grad
