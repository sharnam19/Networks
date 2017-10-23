from util.layer import *
from util.util import *
from util.activations import *
from util.normalization import *
from util.convolution import *
from util.loss import *
from descent.descent import *
import numpy as np
import copy

def regularization(weight,params):
    reg_type = params.get('reg_type')
    if reg_type is not 'None':
        reg = params.get('reg',0.0)
        if reg_type == 'L2':
            return reg*np.sum(np.square(weight))
        elif reg_type == 'L1':
            return reg*np.sum(np.abs(weight))
        else:
            return 0.0
    else:
        return 0.0
    
class Padding():
    
    def __init__(self,padding_h,padding_w):
        self.padding = (padding_h,padding_w)
    
    def forward(self,X):
        return padding_forward(X,self.padding[0],self.padding[1])
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut):
        return padding_backward(dOut,self.padding)
    
class Pooling():
    
    def __init__(self,pooling_params):
        self.pooling_params = pooling_params
        self.cache = None
        
    def forward(self,X):
        out,self.cache = max_pooling_forward(X,self.pooling_params)
        return out
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut):
        out =  max_pooling_backward(dOut,self.cache)
        self.cache = None
        return out

class Convolution():
    
    def __init__(self,w,b,convolution_params,update_params):
        self.w = w
        self.b = b
        self.convolution_params = convolution_params
        self.wparams=copy.deepcopy(update_params)
        self.bparams=copy.deepcopy(update_params)
        
    def forward(self,X):
        out,self.cache = convolve_forward_fast(X,self.w,self.b,self.convolution_params)
        return out
    
    def loss_reg(self):
        return regularization(self.w,self.wparams)
    
    def backprop(self,dOut):
        dx,dw,db = convolve_backward_fast(dOut,self.cache)
        update_weight(self.w,dw,self.wparams,regularization=True)
        update_weight(self.b,db,self.bparams)
        self.cache = None
        return dx

class Relu():
    
    def __init__(self):
        self.cache = None
    
    def forward(self,X):
        out,self.cache = relu_forward(X)
        return out
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut):
        out= relu_backward(dOut,self.cache)
        self.cache = None
        return out
    
class Sigmoid():
    
    def __init__(self):
        self.cache = None
    
    def forward(self,X):
        out,self.cache = sigmoid_forward(X)
        return out
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut):
        out= sigmoid_backward(dOut,self.cache)
        self.cache = None
        return out
    
class Tanh():
    
    def __init__(self):
        self.cache = None
    
    def forward(self,X):
        out,self.cache = tanh_forward(X)
        return out
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut):
        out = tanh_backward(dOut,self.cache)
        self.cache = None
        return out

class LeakyRelu():
    
    def __init__(self):
        self.cache = None
    
    def forward(self,X):
        out,self.cache = leaky_relu_forward(X)
        return out
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut):
        out = leaky_relu_backward(dOut,self.cache)
        self.cache = None
        return out
    
class Affine():
    
    def __init__(self,w,b,update_params):
        self.cache = None
        self.w = w
        self.b = b
        self.wparams = copy.deepcopy(update_params)
        self.bparams = copy.deepcopy(update_params)
        
    def forward(self,X):
        out, self.cache = affine_forward(X,self.w,self.b)
        return out
    
    def loss_reg(self):
        return regularization(self.w,self.wparams)
    
    def backprop(self,dOut):
        dx,dw,db = affine_backward(dOut,self.cache)
        update_weight(self.w,dw,self.wparams,regularization=True)
        update_weight(self.b,db,self.bparams)
        self.cache = None
        return dx

class Flatten():
    
    def __init__(self):
        self.cache = None
    
    def forward(self,X):
        out,self.cache = flatten_forward(X)
        return out
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut):
        out= flatten_backward(dOut,self.cache)
        self.cache = None
        return out
    

class Softmax():
    
    def __init__(self):
        self.dx= None
    
    def forward(self,X,y=None):
        if y is None:
            scores = softmax_loss(X)
            return scores
        else:
            scores,loss,self.dx = softmax_loss(X,y)
            return scores,loss
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut=None):
        return self.dx
    
    def accuracy(self,scores,y):
        return 1.0*np.sum(np.argmax(scores,axis=1)==y)/y.shape[0]
    
class SVM():

    def __init__(self):
        self.dx= None
    
    def forward(self,X,y=None):
        if y is None:
            scores = softmax_loss(X)
            return scores
        else:
            scores,loss,self.dx = svm_loss(X,y)
            return scores,loss
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut=None):
        return self.dx
    
    def accuracy(self,scores,y):
        return 1.0*np.sum(np.argmax(scores,axis=1)==y)/y.shape[0]

def MSE():
    
    def __init__(self):
        self.dx = None
    
    def forward(self,X,y=None):
        if y is None:
            scores = mse_loss(X)
            return scores
        else:
            scores,loss,self.dx = mse_loss(X,y)
            return scores,loss
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut=None):
        return self.dx
    
    def accuracy(self,scores,y):
        return rel_error(scores,y)

class CrossEntropy():
    
    def __init__(self):
        self.dx = None
    
    def forward(self,X,y=None):
        if y is None:
            scores = cross_entropy_loss(X)
            return scores
        else:
            scores,loss,self.dx = cross_entropy_loss(X,y)
            return scores,loss
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut=None):
        return self.dx
    
    def accuracy(self,scores,y):
        return 1.0*np.sum(np.round(scores,0)==y)/y.shape[0]

    
class BatchNormalization(object):
    
    def __init__(self,gamma,beta,params,update_params):
        self.gamma = gamma
        self.beta = beta
        self.params = params
        self.gamma_update_params = copy.deepcopy(update_params)
        self.beta_update_params  = copy.deepcopy(update_params)
        self.cache = None
        
    def forward(self,X):
        out,self.cache = batch_normalization_forward(X,self.gamma,self.beta,self.params)
        return out
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut):
        dx,dgamma,dbeta = batch_normalization_backward(dOut,self.cache)
        update_weight(self.gamma,dgamma,self.gamma_update_params)
        update_weight(self.beta,dbeta,self.beta_update_params)
        self.cache = None
        return dx

class SpatialBatchNormalization(object):
    
    def __init__(self,gamma,beta,params,update_params):
        self.gamma = gamma
        self.beta = beta
        self.params = params
        self.gamma_update_params = copy.deepcopy(update_params)
        self.beta_update_params = copy.deepcopy(update_params)
        self.cache = None
    
    def forward(self,X):
        out,self.cache = spatial_batch_forward(X,self.gamma,self.beta,self.params)
        return out
    
    def loss_reg(self):
        return 0.0
    
    def backprop(self,dOut):
        dx,dgamma,dbeta = spatial_batch_backward(dOut,self.cache)
        update_weight(self.gamma,dgamma,self.gamma_update_params)
        update_weight(self.beta,dbeta,self.beta_update_params)
        self.cache = None
        return dx        
