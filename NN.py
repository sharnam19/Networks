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
            batch_params=None,
            output=None,
            initialization="normal",mean=0,shift=0.01):
        
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
            
            self.layers.append(padding_forward)
            self.back_layers.append(padding_backward)
            
            shape = self.out_shape[-1]
            self.params.append([padding_h,padding_v])
            self.out_shape((shape[0],shape[1],2*padding_h+shape[2],2*padding_v+shape[3]))
            
        elif layer_name == "pooling" and outshape == 4:
            
            self.layers.append(max_pooling_forward)
            self.back_layers.append(max_pooling_backward)
            
            Ph = pooling_params.get('pooling_height',2)
            Pw = pooling_params.get('pooling_width',2)
            PSH = pooling_params.get('pooling_stride_height',2)
            PSW = pooling_params.get('pooling_stride_width',2)
    
            N,C,H,W = self.out_shape[-1]
            Hout = (H-Ph)//PSH + 1
            Wout = (W-Pw)//PSW + 1
            
            self.params.append([pooling_params])
            self.out_shape.append((N,C,Hout,Wout))
    
        elif layer_name == "convolution" and outshape == 4:
            
            N,C,H,W = self.out_shape[-1]
            S = convolution_params.get('stride',1)
            Hout = abs(H-kernel_h)//S + 1
            Wout = abs(W-kernel_w)//S + 1
        
            self.layers.append(convolve_forward_fast)
            self.back_layers.append(convolve_backward_fast)
        
            W = initializer(mean,shift,(num_kernels,C,kernel_h,kernel_w))
            b = np.zeros((num_kernels,))
            self.params.append([W,b,convolution_params])
            
            self.out_shape.append((N,D,Hout,Wout))
            
        elif layer_name == "softmax" and outshape==2:
            
            self.layers.append(softmax_loss)
            self.back_layers.append([])
            
            self.params.append([output])
            self.out_shape.append((1))
            
        elif layer_name == "svm" and outshape==2:
            self.layers.append(svm_loss)
            self.back_layers.append([])
            
            self.params.append([output])
            self.out_shape.append((1))
        elif layer_name == "batch_normalization" and outshape==2:
            
            self.layers.append(batch_normalization_forward)
            self.back_layers.append(batch_normalization_backward)
            
            shape = self.out_shape[-1]
            D = shape[1]
            
            gamma = np.ones((D,))
            beta = np.zeros((D,))
            self.params.append([gamma,beta,batch_params])
            
            self.out_shape.append(shape)
        elif layer__name == "spatial_batch" and outshape==4:
            
            self.layers.append(spatial_batch_forward)
            self.back_layers.append(spatial_batch_backward)
            
            shape = self.out_shape[-1]
            gamma = np.ones((shape[1],))
            beta = np.zeros((shape[1]))
            
            self.params.append([gamma,beta,batch_params])
            self.out_shape.append(shape)
        else:
            print "Check Shapes"