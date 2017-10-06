from util.layer import *
from util.util import *
from util.activations import *
from util.normalization import *
from util.convolution import *
from util.loss import *
from descent.descent import *
import numpy as np
import copy
class Padding():
    
    def __init__(self,padding_h,padding_w):
        self.padding = (padding_h,padding_w)
    
    def forward(X):
        return padding_forward(X,self.padding[0],self.padding[1])
    
    def backprop(dOut):
        return padding_backward(dOut,self.padding)
    
class Pooling():
    
    def __init__(self,pooling_params):
        self.pooling_params = pooling_params
        self.cache = None
        
    def forward(X):
        out,self.cache = max_pooling_forward(X,self.pooling_params)
        return out
    
    def backprop(dOut):
        return max_pooling_backward(dOut,self.cache)

class Convolution():
    
    def __init__(self,w,b,convolution_params,update_params):
        self.w = w
        self.b = b
        self.convolution_params = convolution_params
        self.wparams=copy.deepcopy(update_params)
        self.bparams=copy.deepcopy(update_params)
        
    def forward(X):
        out,self.cache = convolve_forward_fast(X,self.w,self.b,self.convolution_params)
        return out
    
    def backprop(dOut):
        dx,dw,db = convolve_backward_fast(dOut,self.cache)
        update_weight(self.w,dw,self.wparams)
        update_weight(self.b,db,self.bparams)
        return dx

class Relu():
    
    def __init__(self):
        self.cache = None
    
    def forward(X):
        out,self.cache = relu_forward(X)
        return out
    
    def backprop(dOut):
        return relu_backward(dOut,self.cache)
    
class Sigmoid():
    
    def __init__(self):
        self.cache = None
    
    def forward(X):
        out,self.cache = sigmoid_forward(X)
        return out
    
    def backprop(dOut):
        return sigmoid_backward(dOut,self.cache)
    
class Tanh():
    
    def __init__(self):
        self.cache = None
    
    def forward(X):
        out,self.cache = tanh_forward(X)
        return out
    
    def backprop(dOut):
        return tanh_backward(dOut,self.cache)

class LeakyRelu():
    
    def __init__(self):
        self.cache = None
    
    def forward(X):
        out,self.cache = leaky_relu_forward(X)
        return out
    
    def backprop(dOut):
        return leaky_relu_backward(dOut,self.cache)
    
class Affine():
    
    def __init__(self,w,b,update_params):
        self.cache = None
        self.w = w
        self.b = b
        self.wparams = copy.deepcopy(update_params)
        self.bparams = copy.deepcopy(update_params)
        
    def forward(X):
        out, self.cache = affine_forward(X,self.w,self.b)
        return out
    
    def backprop(dOut):
        dx,dw,db = affine_backward(dOut,self.cache)
        update_weight(self.w,dw,self.wparams)
        update_weight(self.b,db,self.bparams)
        return dx

class Flatten():
    
    def __init__(self):
        self.cache = None
    
    def forward(X):
        out,cache = flatten_forward(X)
        return out
    
    def backprop(dOut):
        return flatten_backprop(dOut,self.cache)
    

class Softmax():
    
    def __init__(self):
        self.dx= None
    
    def forward(X,y=None):
        loss,self.dx = softmax_loss(X,y)
        return loss
    
    def backprop(dOut=None):
        return self.dx

class SVM():

    def __init__(self):
        self.dx= None
    
    def forward(X,y=None):
        loss,self.dx = svm_loss(X,y)
        return loss
    
    def backprop(dOut=None):
        return self.dx

class BatchNormalization():
    
    def __init__(self,gamma,beta,params,update_params):
        self.gamma = gamma
        self.beta = beta
        self.params = params
        self.gamma_update_params = copy.deepcopy(update_params)
        self.beta_update_params  = copy.deepcopy(update_params)
        self.cache = None
        
    def forward(X):
        out,self.cache = batch_normalization_forward(X,self.gamma,self.beta,self.params)
        return out
    
    def backprop(dOut):
        dx,dgamma,dbeta = batch_normalization_backward(dOut,self.cache)
        update_weight(self.gamma,dgamma,self.gamma_update_params)
        update_weight(self.beta,dbeta,self.beta_update_params)
        return dx

class SpatialBatchNormalization():
    
    def __init__(self,gamma,beta,params,update_params):
        self.gamma = gamma
        self.beta = beta
        self.params = params
        self.gamma_update_params = copy.deepcopy(update_params)
        self.beta_update_params = copy.deepcopy(update_params)
        self.cache = None
    
    def forward(X):
        out,self.cache = spatial_batch_forward(X,self.gamma,self.beta,self.params)
        return out
    
    def backprop(dOut):
        dx,dgamma,dbeta = spatial_batch_backward(dOut,self.cache)
        update_weight(self.gamma,dgamma,self.gamma_update_params)
        update_weight(self.beta,dbeta,self.beta_update_params)
        return dx        