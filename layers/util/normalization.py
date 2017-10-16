import numpy as np

def batch_normalization_forward(x, gamma,beta,params):
    """
    Input:
        x: Array of shape (N,D)
        gamma : Scale Parameter of shape (D,)
        beta : Shift Parameter of shape (D,)
        params:
            'mode' : 'train' or 'test'
            'running_var' : Current value of exponential decayed variance over train set
            'running_mean': Current value of exponential decayed mean over train set
            'momentum' : Rate at which mean and variance should be decayed
            'eps' : Epsilon value to prevent shooting of variance
    """
    mode = params.get('mode','train')
    momentum = params.get('momentum',0.9)
    running_mean = params.get('running_mean',np.zeros(x.shape[1]))
    running_var  = params.get('running_var',np.zeros(x.shape[1]))
    eps = params.get('eps',1e-8)
    
    out, cache = None, None
    if mode == "train":
        mu = np.mean(x,axis=0)
        t  = x - mu
        f  = t**2
        u = np.mean(f,axis=0)
        k = np.power(u+eps,0.5)
        p = 1./k
        q = t*p
        r = q*gamma
        out = r + beta
        params['running_mean']  = momentum*running_mean + (1-momentum)*mu
        params['running_var']   = momentum*running_var + (1-momentum)*u
        cache = (x,gamma,q,t,p)
    else:
        t = x - running_mean
        u = running_var
        k = np.power(u + eps,0.5)
        p = 1/k
        q = t*p
        r = q*gamma
        out = beta + r
        
    return out,cache

def batch_normalization_backward(dOut,cache):
    """
        Input
            dOut : Upstream Gradients of shape (N,D)
            cache : (x,gamma,beta,params,q,t,p,k)
        Output
            dx,dgamma,dbeta of same shape as x,gamma,beta
    """
    x, gamma, q, t, p = cache
    N, D = x.shape
    dbeta  = np.sum(dOut,axis=0)
    dgamma = np.sum(dOut*q,axis=0)
    dr = dOut
    dq = gamma*dOut
    dp = np.sum(t*dq,axis=0)
    dt1 = p*dq
    dk = -(p**2)*dp
    du = 0.5*p*dk
    df = np.ones((N,D))*du*1./N
    dt2 = 2*t*df
    dt = dt1+dt2
    dx1 = dt
    dmu = -np.sum(dx1,axis=0)
    dx2 = 1./N*np.ones((N,D))*dmu
    dx = dx1+dx2
    
    return dx,dgamma,dbeta

def spatial_batch_forward(x,gamma,beta,params):
    """
    Input:
        x: Array of shape (N,C,H,W)
        gamma : Scale Parameter of shape (C,)
        beta : Shift Parameter of shape (C,)
        params:
            'mode' : 'train' or 'test'
            'running_var' : Current value of exponential decayed variance over train set
            'running_mean': Current value of exponential decayed mean over train set
            'momentum' : Rate at which mean and variance should be decayed
            'eps' : Epsilon value to prevent shooting of variance
    """
    N,C,H,W=x.shape
    #Reshapes array to N,H,W,C
    reshaped_x = np.swapaxes(np.swapaxes(x,1,2),2,3)
    reshaped_x = reshaped_x.reshape(-1,C)
    
    out,cache = batch_normalization_forward(reshaped_x,gamma,beta,params)
    out = out.reshape(N,H,W,C)
    out = np.swapaxes(np.swapaxes(out,2,3),2,1)
    return out,cache

def spatial_batch_backward(dOut,cache):
    """
        Input
            dOut : Upstream Gradients of shape (N,C,H,W)
            cache : (x,gamma,beta,params,q,t,p,k)
        Output
            dx,dgamma,dbeta of same shape as x,gamma,beta
    """
    N,C,H,W = dOut.shape
    reshaped_dOut = np.swapaxes(np.swapaxes(dOut,1,2),2,3)
    reshaped_dOut = reshaped_dOut.reshape(-1,C)
    dx,dgamma,dbeta = batch_normalization_backward(reshaped_dOut,cache)
    dx = dx.reshape(N,H,W,C)
    dx = np.swapaxes(np.swapaxes(dx,2,3),1,2)
    return dx,dgamma,dbeta