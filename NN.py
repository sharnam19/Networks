from util.layer import *
from util.util import *
from util.activations import *
from util.normalization import *
from util.convolution import *
from util.loss import *
from operator import mul

class NN:
    
    def __init__(self,input_shape):
        self.J=[]
        self.layers=[]
        self.back_layers=[]
        self.params=[]
        self.out_shape=[input_shape]
        self.cache=[]
        
    def initializer(mean=0,shift=0.01,shape=None,initialization="normal"):
        if initialization == "normal":
            init = shift*np.random.standard_normal(shape)+mean
        elif initialization == "xavier":
            init = np.random.standard_normal(shape)/(shape[0]**0.5)
        elif initialization == "xavier2":
            init = np.random.standard_normal(shape)/((shape[0]/2)**0.5)
        return init
    
    def add(self,layer_name,affine_out=None,
            padding_h=None,padding_v=None,
            pooling_params=None,
            num_kernels=None,kernel_h=None,kernel_w=None,convolution_params=None,
            batch_params=None,initialization="normal",mean=0,shift=0.01):
        
        outshape = len(self.out_shape[-1])
        
        if layer_name == "affine" and outshape==2:
        
            self.layers.append(affine_forward)
            self.back_layers.append(affine_backward)
            N,D = self.out_shape[-1]
            W = initializer(mean,shift,(D,affine_out),initialization=initialization)
            b = np.zeros(affine_out,)
            self.params.append([W,b])
            self.out_shape.append((N,affine_out))
            
        elif layer_name == "flatten" and outshape>2:
            
            self.layers.append(flatten_forward)
            self.back_layers.append(flatten_backward)
            shape = self.out_shape[-1]
            self.params.append([])
            self.out_shape.append((shape[0],reduce(mul,shape[1:],1)))
        
        elif layer_name == "relu":
        
            self.layers.append(relu_forward)
            self.back_layers.append(relu_backward)
            shape = self.out_shape[-1]
            self.params.append([])
            self.out_shape.append(shape)
        
        elif layer_name == "sigmoid":
            
            self.layers.append(sigmoid_forward)
            self.back_layers.append(sigmoid_backward)
            
            shape = self.out_shape[-1]
            self.params.append([])
            self.out_shape.append(shape)
            
        elif layer_name == "tanh":
            
            self.layers.append(tanh_forward)
            self.back_layers.append(tanh_backward)
            
            shape = self.out_shape[-1]
            self.params.append([])
            self.out_shape.append(shape)
            
        elif layer_name == "leaky_relu":
            
            self.layers.append(leaky_relu_forward)
            self.back_layers.append(leaky_relu_backward)
            
            shape = self.out_shape[-1]
            self.params.append([])
            self.out_shape.append(shape)
            
        elif layer_name == "padding" and outshape == 4:
            pass
        elif layer_name == "pooling" and outshape == 4:
            pass
        elif layer_name == "convolution" and outshape == 4:
            pass
        elif layer_name == "softmax" and outshape==2:
            pass
        elif layer_name == "svm" and outshape==2:
            pass
        else:
            print "Check Shapes"