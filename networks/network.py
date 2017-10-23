from layers.layer import *
from operator import mul
import matplotlib.pyplot as plt
import pickle
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()
class network:
    
    def __init__(self,input_shape,update_params,initialization="normal"):
        self.J=[]
        self.layers=[]
        self.accuracies=[]
        self.out_shape=[input_shape]
        self.update_params=update_params
        self.initialization = initialization
        self.normalization_layers=[]
        
    def initializer(self,mean=0,shift=0.01,shape=None,initialization="normal"):
        if initialization == "normal":
            init = shift*np.random.standard_normal(shape)+mean
        elif initialization == "xavier":
            init = np.random.standard_normal(shape)/(shape[0]**0.5)
        elif initialization == "xavier2":
            if len(shape)==2:
                init = np.random.standard_normal(shape)/((shape[0]/2.)**0.5)
            elif len(shape)==4:
                init = np.random.standard_normal(shape)/((reduce(mul,shape[1:],1)/2.)**0.5)
        return init

    def add(self,layer_name,affine_out=None,
            padding_h=None,padding_w=None,
            pooling_params=None,
            num_kernels=None,kernel_h=None,kernel_w=None,convolution_params=None,
            batch_params=None,
            output=None,
            mean=0,shift=0.01):
        
        outshape = len(self.out_shape[-1])
        
        if layer_name == "affine" and outshape==2:
            
            N,D = self.out_shape[-1]
            W = self.initializer(mean,shift,(D,affine_out),initialization=self.initialization)
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
        
            W = self.initializer(mean,shift,(num_kernels,C,kernel_h,kernel_w),initialization=self.initialization)
            b = np.zeros((num_kernels,))
            self.layers.append(Convolution(W,b,convolution_params,self.update_params))
            self.out_shape.append((N,num_kernels,Hout,Wout))
            
        elif layer_name == "softmax" and outshape==2:
            
            self.layers.append(Softmax())
            self.out_shape.append((1))
            
        elif layer_name == "svm" and outshape==2:
            
            self.layers.append(SVM())
            self.out_shape.append((1))
            
        elif layer_name == "mse" and outshape == 2 and self.out_shape[-1][1]==1:
            
            self.layers.append(MSE())
            self.out_shape.append((1))
        
        elif layer_name =="cross_entropy" and outshape==2:

            self.layers.append(CrossEntropy())
            self.out_shape.append((1))
            
        elif layer_name == "batch_normalization" and outshape==2:
            
            shape = self.out_shape[-1]
            D = shape[1]
            
            gamma = np.ones((D,))
            beta = np.zeros((D,))
            self.layers.append(BatchNormalization(gamma,beta,batch_params,self.update_params))
            self.normalization_layers.append(self.layers[-1])
            self.out_shape.append(shape)
            
        elif layer_name == "spatial_batch" and outshape==4:
            
            shape = self.out_shape[-1]
            gamma = np.ones((shape[1],))
            beta = np.zeros((shape[1],))
            self.layers.append(SpatialBatchNormalization(gamma,beta,batch_params,self.update_params))
            self.normalization_layers.append(self.layers[-1])
            self.out_shape.append(shape)
            
        else:
            print "Check Shapes"
            raise NotImplementedError
    
    
    def train(self,X,y):
        batch_size = self.out_shape[0][0]
        acc, cost = self.test(X[:batch_size],y[:batch_size])
        self.accuracies.append(acc)
        self.J.append(cost)
        print("Initial Cost :"+str(cost))
        print("Initial Accuracy :"+str(acc))
        
        for i in range(self.update_params['epoch']):
            sample  = np.random.randint(0,X.shape[0],(self.out_shape[0][0],))
            inp = X[sample]
            loss = 0.0
            for layer in self.layers[:-1]:
                inp = layer.forward(inp)
                loss += layer.loss_reg()
            
            scores,inp = self.layers[-1].forward(inp,y[sample])
            
            for layer in self.layers[::-1]:
                inp = layer.backprop(inp)
            
            acc,cost = self.test(X[sample],y[sample])
            self.accuracies.append(acc)
            self.J.append(cost)
            print("Cost at Iteration "+str(i)+" : "+str(cost))
            print("Accuracy at Iteration "+str(i)+" : "+str(acc))
                  
    def test(self,X,y):
        accuracies = []
        loss = []
        for normalization_layer in self.normalization_layers:
            normalization_layer.params['mode']='test'
            
        batch_size = self.out_shape[0][0]
        for i in range(0,X.shape[0],batch_size):
            end = min(i+batch_size,X.shape[0])
            batch_acc,batch_loss = self.batch_test(X[i:end],y[i:end])
            accuracies.append(batch_acc)
            loss.append(batch_loss)
        
        accuracies = np.array(accuracies)
        loss = np.array(loss)
        for normalization_layer in self.normalization_layers:
            normalization_layer.params['mode']='train'
            
        return np.mean(accuracies),np.mean(loss)
        
    def batch_test(self,X,y):
        loss = 0.0
        inp = X
        for layer in self.layers[:-1]:
            inp = layer.forward(inp)
            loss += layer.loss_reg()
        
        scores,inp = self.layers[-1].forward(inp,y)
        return self.layers[-1].accuracy(scores,y),inp+loss
    
    def predict(self,X):
        inp = X
        for normalization_layer in self.normalization_layers:
            normalization_layer.params["mode"]="test"
        
        for layer in self.layers:
            inp = layer.forward(inp)
        
        for normalization_layer in self.normalization_layers:
            normalization_layer.params["mode"]="train"
            
        return np.argmax(inp,axis=1)
    
    def save(self,filename):
        outfile = open('models/'+filename, 'wb')
        enc = jsonpickle.encode(self)
        json.dump(enc,outfile)
    
    @staticmethod
    def load(filename):
        infile = open('models/'+filename,'rb')
        return jsonpickle.decode(json.load(infile))
    
    def plot(self):
        plt.plot(self.J[1:])
        plt.title("Loss VS Iteration")
        plt.show()
