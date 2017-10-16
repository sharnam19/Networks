from NN import NN
import json
import numpy as np
if __name__== "__main__":
    model = NN(input_shape=(64,1,50,50),update_params={'alpha':1e-3,'method':'adam','epoch':100,'offset':1e-7,'reg':0.01,'reg_type':'L2'},initialization="xavier2")
    model.add("padding",padding_h=2,padding_w=2)
    model.add("convolution",num_kernels=64,kernel_h=3,kernel_w=3,convolution_params={"stride":1})
    model.add("pooling",pooling_params={"pooling_height":2,"pooling_width":2,
                                       "pooling_stride_height":2,"pooling_stride_width":2})
    model.add("relu")
    model.add("convolution",num_kernels=128,kernel_h=3,kernel_w=3,convolution_params={"stride":1})
    model.add("pooling",pooling_params={"pooling_height":2,"pooling_width":2,
                                       "pooling_stride_height":2,"pooling_stride_width":2})
    
    model.add("relu")
    model.add("flatten")
    model.add("affine",affine_out=128)
    model.add("affine",affine_out=64)
    model.add("affine",affine_out=16)
    model.add("affine",affine_out=5)
    model.add("softmax")
    
    data = json.load(open("data/data.json","rb"))
    trainX = np.array(data['trainX'])
    trainY = np.array(data['trainY'],dtype=np.int32)
    
    validX = np.array(data['validX'])
    validY = np.array(data['validY'],dtype=np.int32)
    
    testX = np.array(data['testX'])
    testY = np.array(data['testY'],dtype=np.int32)
    model.train(trainX,trainY)
    model.save("model1.pkl")