# from layers.util.layer import *
# from layers.util.util import *
# from layers.util.activations import *
# from layers.util.normalization import *
# from layers.util.convolution import *
# from layers.util.loss import *
from layers.layer import *
from operator import mul

class NN:
    
    def __init__(self,input_shape,update_params):
        self.J=[]
        self.layers=[]
        self.out_shape=[input_shape]
        self.update_params=update_params
        
    def initializer(self,mean=0,shift=0.01,shape=None,initialization="normal"):
        if initialization == "normal":
            init = shift*np.random.standard_normal(shape)+mean
        elif initialization == "xavier":
            init = np.random.standard_normal(shape)/(shape[0]**0.5)
        elif initialization == "xavier2":
            init = np.random.standard_normal(shape)/((shape[0]/2)**0.5)
        return init

    def add(self,layer_name,affine_out=None,
            padding_h=None,padding_w=None,
            pooling_params=None,
            num_kernels=None,kernel_h=None,kernel_w=None,convolution_params=None,
            batch_params=None,
            output=None,
            initialization="normal",mean=0,shift=0.01):
        
        outshape = len(self.out_shape[-1])
        
        if layer_name == "affine" and outshape==2:
            
            N,D = self.out_shape[-1]
            W = self.initializer(mean,shift,(D,affine_out),initialization=initialization)
            b = np.zeros(affine_out,)
            self.layers.append(Affine(W,b,self.update_params))
            self.out_shape.append((N,affine_out))
            
        elif layer_name == "flatten" and outshape>2:
        
            self.layers.append(Flatten())
            shape = self.out_shape[-1]
            self.out_shape.append((shape[0],reduce(mul,shape[1:],1)))
        
        elif layer_name == "relu":
            
            self.layers.append(Relu())
            shape = self.out_shape[-1]
            self.out_shape.append(shape)
        
        elif layer_name == "sigmoid":
            
            self.layers.append(Sigmoid())
            shape = self.out_shape[-1]
            self.out_shape.append(shape)
            
        elif layer_name == "tanh":
            
            self.layers.append(Tanh())
            shape = self.out_shape[-1]
            self.out_shape.append(shape)
            
        elif layer_name == "leaky_relu":
            
            self.layers.append(LeakyRelu())
            shape = self.out_shape[-1]
            self.out_shape.append(shape)
            
        elif layer_name == "padding" and outshape == 4:
            
            shape = self.out_shape[-1]
            self.layers.append(Padding(padding_h,padding_w))
            self.out_shape.append((shape[0],shape[1],2*padding_h+shape[2],2*padding_w+shape[3]))
            
        elif layer_name == "pooling" and outshape == 4:
            
            self.layers.append(Pooling(pooling_params))
            
            Ph = pooling_params.get('pooling_height',2)
            Pw = pooling_params.get('pooling_width',2)
            PSH = pooling_params.get('pooling_stride_height',2)
            PSW = pooling_params.get('pooling_stride_width',2)
    
            N,C,H,W = self.out_shape[-1]
            Hout = (H-Ph)//PSH + 1
            Wout = (W-Pw)//PSW + 1
            
            self.out_shape.append((N,C,Hout,Wout))
    
        elif layer_name == "convolution" and outshape == 4:
            
            N,C,H,W = self.out_shape[-1]
            S = convolution_params.get('stride',1)
            Hout = abs(H-kernel_h)//S + 1
            Wout = abs(W-kernel_w)//S + 1
        
            W = self.initializer(mean,shift,(num_kernels,C,kernel_h,kernel_w))
            b = np.zeros((num_kernels,))
            self.layers.append(Convolution(W,b,convolution_params,self.update_params))
            self.out_shape.append((N,num_kernels,Hout,Wout))
            
        elif layer_name == "softmax" and outshape==2:
            
            self.layers.append(Softmax())
            self.out_shape.append((1))
            
        elif layer_name == "svm" and outshape==2:
            
            self.layers.append(SVM())
            self.out_shape.append((1))
            
        elif layer_name == "batch_normalization" and outshape==2:
            
            shape = self.out_shape[-1]
            D = shape[1]
            
            gamma = np.ones((D,))
            beta = np.zeros((D,))
            self.layers.append(BatchNormalization(gamma,beta,batch_params,self.update_params))
            self.out_shape.append(shape)
        elif layer_name == "spatial_batch" and outshape==4:
            
            shape = self.out_shape[-1]
            gamma = np.ones((shape[1],))
            beta = np.zeros((shape[1],))
            self.layers.append(SpatialBatchNormalization(gamma,beta,batch_params,self.update_params))
            self.out_shape.append(shape)
        else:
            print "Check Shapes"
            raise NotImplementedError
    
    
    def train(X,y):
        for i in range(self.update_params['epoch']):
            inp = X
            for layer in self.layers[:-1]:
                inp = layer.forward(inp)
                
            inp = self.layers[-1].forward(inp,y)
            
            self.J.append(inp)
            for layer in self.layers[::-1]:
                inp = layer.backprop(inp)
        
if __name__== "__main__":
    model = NN(input_shape=(64,3,32,32),update_params={'alpha':1e-3,'method':'gd','epoch':1000})
    model.add("padding",padding_h=2,padding_w=2)
    model.add("convolution",num_kernels=128,kernel_h=3,kernel_w=3,convolution_params={"stride":1})
    model.add("flatten")
    model.add("affine",affine_out=10)
    model.add("softmax")
    print(model.layers)
    print(model.out_shape)
            