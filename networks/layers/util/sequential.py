import numpy as np
def rnn_step(x, h_prev, Wx, Wh, b):
    """
        Input:
            x : Input Sequence of shape (N,D)
            h_prev : Previous Hidden State (N,H)
            Wx : Input to Hidden Weight of shape (D,H)
            Wh : Hidden to Hidden Weight of shape (H,H)
            b :  Bias of shape (H,)
        Output:
            h_next : Hidden State at Next Time Step of shape (N,H)
            cache : Cached Values for Backprop 
    """
    
    h_next = np.tanh(x.dot(Wx)+h_prev.dot(Wh) + b[np.newaxis,:])
    cache = (x,h_prev,Wx,Wh,h_next)
    return h_next,cache

def rnn_step_backward(dOut,cache):
    """
        Input
            dOut: Upstream Gradients wrt h (N,H)
            cache : Cached Values useful for backprop
        Output:
            dx: Gradients wrt x
            dh_prev : Gradients wrt h_prev
            dWx : Gradients wrt Wx
            dWh : Gradients wrt Wh
            db : Gradients wrt b
    """
    x,h_prev,Wx,Wh,h_next = cache
    dSq = (1-np.square(h_next))*dOut
    
    dx = dSq.dot(Wx.T)
    dWx = x.T.dot(dSq)
    dh_prev = dSq.dot(Wh.T)
    dWh = h_prev.T.dot(dSq)
    db = np.sum(dSq,axis=0)
    
    return dx,dh_prev,dWx,dWh,db

def rnn_forward(x,h0,Wx,Wh,b):
    """
        Input:
            x: Input of shape (N,T,D)
            h0: Initial State of shape (N,H)
            Wx : Input to Hidden Weights (D,H)
            Wh : Hidden to Hidden Weights (H,H)
            b : Bias of shape (H,)
    """
    prev_h = h0
    caches=[]
    N,T,D = x.shape
    H = h0.shape[1]
    h = np.zeros((N,T,H))
    for t in range(T):
        prev_h,temp_cache = rnn_step(x[:,t,:],prev_h,Wx,Wh,b)
        h[:,t,:]=prev_h
        caches.append(temp_cache)
    return h,caches

def rnn_backward(dh,caches):
    N,T,H = dh.shape
    D = caches[0][0].shape[1]
    dx = np.zeros((N,T,D))
    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    db = np.zeros((H,))
    dprev_h=np.zeros((N,H))
    for t in range(T)[::-1]:
        dx[:,t,:],dprev_h,dWx_t,dWh_t,db_t = rnn_step_backward(dprev_h+dh[:,t,:],caches[t])
        dWx+=dWx_t
        dWh+=dWh_t
        db+=db_t
    
    return dx,dprev_h,dWx,dWh,db