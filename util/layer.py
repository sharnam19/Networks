import numpy as np
from scipy import signal

def affine_forward(x, w, b):
    """
        Input
        x: Input of shape (N,D)
        w: Weights of shape (D,H)
        b: Biases of shape (H,)

        Output(out,cache)
        Returns out of shape (N,H)

    """
    out = x.dot(w)+b
    cache = (x,w,b)
    return out,cache

def affine_backward(dOut,cache):
    """
        Input:
        dOut: Upstream gradient of shape (N,H)
        Returns Derivative wrt the inputs x,w,b i.e
        dx,dw,db
    """
    x,w,b = cache
    dx = np.dot(dOut,w.T)
    dw = np.dot(x.T,dOut)
    db = np.sum(dOut,axis=0)
        
    return dx,dw,db

def flatten_forward(x):
    """
        Input:
            x of any shape (N,D1,D2,D3,.....Dn)
        Output:
            out : of shape (N,D1*D2*D3*....Dn)
            cache: consisting of shape
    """
    shape = x.shape
    out = x.reshape(shape[0],-1)
    cache = (shape)
    return out,cache

def flatten_backward(dOut,cache):
    """
    Input:
        dOut: Upstream gradients in shape (N,D1*D2*D3*.....Dn)
        cache: (shape of input tensor)
    Output:
        dx: dOut reshaped in shape of x
    """
    (shape)=cache
    dx = dOut.reshape(shape)
    return dx

def batch_normalization_forward(x, gamma, beta, params):
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

def padding_forward(x, Ph, Pv):
    """
        Input:
            x : of any shape
            Ph: Num of horizontal Padding on top and bottom
            Pv: Num of vertical Padding on sides
            
        Output:
            out : Padded input with Ph and Pv layers of padding
            cache:(Ph,Pv)
    """
    shapes = list(x.shape)
    shapes[-2]+=2*Ph
    shapes[-1]+=2*Pv
    
    out = np.zeros(shapes)
    out[:,:,Ph:-Ph,Pv:-Pv]=x
    cache = (Ph,Pv)
    return out,cache

def padding_backward(dOut,cache):
    """
        Input:
            dOut: any 4D numpy array i.e the upstream gradient
            cache: Contains Horizontal and Vertical Padding Num
        Output:
            dx : of same shape as x
    """
    Ph, Pv = cache
    dx = dOut[:,:,Ph:-Ph,Pv:-Pv]
    return dx

def max_pooling_forward(x, pooling_params):
    """
        Input:
            x: Image of shape (N,C,H,W)
            pooling_params:
                'pooling_height':Pooling Height
                'pooling_width':Pooling Width
                'pooling_stride_height':Pooling Stride Height
                'pooling_stride_width':Pooling Stride Width
        Output:
            out: Pooled Image (N,C,Hout,Wout)
            cache : (out,x,pooling_params)
    """
    Ph = pooling_params.get('pooling_height',2)
    Pw = pooling_params.get('pooling_width',2)
    PSH = pooling_params.get('pooling_stride_height',2)
    PSW = pooling_params.get('pooling_stride_width',2)
    
    N,C,H,W = x.shape
    Hout = (H-Ph)//PSH + 1
    Wout = (W-Pw)//PSW + 1
    out = np.zeros((N,C,Hout,Wout))
    for h in range(Hout):
        top = h*PSH
        bottom = top + Ph
        for w in range(Wout):
            left = w*PSW
            right = left + Pw
            out[:,:,h,w] = np.max(x[:,:,top:bottom,left:right],axis=(2,3))
    
    cache = (out,x,pooling_params)
    return out,cache

def max_pooling_backward(dOut,cache):
    """
        Input:
            dOut: Upstream Gradient
            cache : (out,x,pooling_params)
        Output:
            dx:Gradient wrt x
    """
    
    out,x,pooling_params = cache
    
    dx=np.zeros_like(x)
    
    Ph = pooling_params.get('pooling_height',2)
    Pw = pooling_params.get('pooling_width',2)
    PSH = pooling_params.get('pooling_stride_height',2)
    PSW = pooling_params.get('pooling_stride_width',2)
    
    N,C,Hout,Wout = dOut.shape
    temp_ones = np.ones((N,C,Ph,Pw))
    for h in range(Hout):
        top = h*PSH
        bottom = top+Ph
        for w in range(Wout):
            left = w*PSW
            right = left+Pw
            mask  = temp_ones*out[:,:,h,w][:,:,np.newaxis,np.newaxis]
            max_to_1 = x[:,:,top:bottom,left:right]==mask
            dx[:,:,top:bottom,left:right]=max_to_1*dOut[:,:,h,w][:,:,np.newaxis,np.newaxis]
    
    return dx

def convolve_forward_naive(x,w,b,params):
    """
        Input:
            x: of shape N,C,H,W
            w: of shape D,C,HH,WW
            b: of shape D,
            params
        Output:
            out : convolved Input
            cache : (x,w,b)
    """
    N,C,H,W = x.shape
    D,_,HH,WW = w.shape
    S = params.get('stride',1)
    
    Hout = (H-HH)//S + 1
    Wout = (W-WW)//S + 1
    
    out = np.zeros((N,D,Hout,Wout))
    reshaped_w = w.reshape(D,-1)
    reshaped_w = np.swapaxes(reshaped_w,0,1)
    
    for h in range(Hout):
        top = h*S
        bottom = top + HH
        for w in range(Wout):
            left = w*S
            right = left + WW
            reshaped_x = x[:,:,top:bottom,left:right].reshape(N,-1)
            out[:,:,h,w] = reshaped_x.dot(reshaped_w)
    
    cache = (x,w,params)
    out += b[np.newaxis,:,np.newaxis,np.newaxis]
    
    return out,cache

def convolve(x,w,params,mode='valid',backprop=False):
    N,C,H,W = x.shape
    D,_,HH,WW = w.shape
    S = params.get('stride',1)
    
    Hout = abs(H-HH)//S + 1
    Wout = abs(W-WW)//S + 1
    
    out = np.zeros((N,D,Hout,Wout))
    if not backprop:
        w = np.flip(np.flip(w,3),2)
        
    Hpadded = H + HH - 1
    Wpadded = W + WW - 1
    fx = np.fft.fft2(x,(Hpadded,Wpadded),axes=(-2,-1))
    fw = np.fft.fft2(w,(Hpadded,Wpadded),axes=(-2,-1))
    
    padH = (Hpadded-Hout)//2
    padW = (Wpadded-Wout)//2
    fz = fx[:,np.newaxis,:,:,:] * fw[np.newaxis,:,:,:,:]

    fz = np.sum(fz,axis=(2))
    out = np.fft.ifft2(fz)
    if mode == 'valid':
        out = out[:,:,padH:-padH,padW:-padW].real
    elif mode == 'full':
        out = out.real
    return out

def convolve_forward_fast(x, w, b, params):
    out = convolve(x,w,params)
    out += b[np.newaxis,:,np.newaxis,np.newaxis]
    
    cache = (x,w,params)
    
    return out,cache

def convolve_backward_fast(dOut,cache):
    """
        Input:
            dOut: Upstream Gradients
            cache : x,w,b
        Output:
            dx:Derivative wrt x
            dw:Derivative wrt w
            db:Derivative wrt b
    """
    x,w,params=cache
    
    N,C,H,W = x.shape
    D,_,HH,WW = w.shape
    
    db = np.sum(dOut,axis=(0,2,3))
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    
    shifted_w = np.swapaxes(w,0,1)
    shifted_d = np.swapaxes(dOut,0,1)
    shifted_d = np.flip((np.flip(shifted_d,3)),2)
    shifted_x = np.swapaxes(x,0,1)
    
    dx = convolve(dOut,shifted_w,params,mode='full',backprop=True)
    dw = convolve(shifted_d,shifted_x,params,mode='valid',backprop=True)
    
    return dx,dw,db
