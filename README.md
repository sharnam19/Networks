# Modular Machine/Deep Learning Library
Machine Learning and Deep Learning Models from Scratch.\
This Library allows users to create **Feed-Forward Neural Networks,\
Convolution Neural Networks, Linear Regression & Logistic Regression**.
Without having to write any **backpropagation** code.

To install the **Networks** Library
<pre>
pip install numpy
pip install scipy
pip install jsonpickle
pip install matplotlib
pip install networks</pre>

# Layers in the Library & their Parameters in Add function

## Activation Layers

### 1. Relu Layer
<pre>No Params</pre>
### 2. Sigmoid Layer
<pre>No Params</pre>
### 3. Tanh Layer
<pre>No Params</pre>
### 4. Leaky Relu Layer
<pre>No Params</pre>

## Normalization Layers

### 1. Batch Normalization Layer
<pre>
batch_params={
  'mode':'train'/'test',
  'momentum':0.9,
  'eps':1e-8
  }</pre>
### 2. Spatial Batch Normalization Layer
<pre>batch_params={
  'mode':'train'/'test',
  'momentum':0.9,
  'eps':1e-8
  }</pre>

## Convolution Layers

### 1. Max Pooling Layer
<pre>
pooling_params={
  'pooling_height':2,
  'pooling_width':2,
  'pooling_stride_height':2,
  'pooling_stride_width':2
}
</pre>
### 2. Convolution Layer
<pre>
num_kernels=64,
kernel_h=3,
kernel_w=3,
convolution_params={
  'stride':1
}
</pre>
### 3. Padding Layer
<pre>
padding_h=2,
padding_w=2
</pre>

## Loss Layers

### 1. Softmax Loss Layer
<pre>No params</pre>
### 2. SVM Loss Layer
<pre>No params</pre>
### 3. Mean Squared Error Layer
<pre>No params</pre>
### Fully Connected Layer

### 1. Affine Layer
<pre>affine_out = 64</pre>
### 2. Flatten Layer
<pre>No params</pre>

# Example Usage
<pre>
from networks.network import network
model = network(input_shape=(64,1,50,50),initialization="xavier2",
update_params={
  'alpha':1e-3,
  'method':'adam',
  'epoch':100,
  'reg':0.01,
  'reg_type':'L2',
  'offset':1e-7
})</pre>

### To Add Padding Layer
<pre>model.add("padding",padding_h=3,padding_w=3)</pre>

### To Add Convolution Layer
<pre>model.add("convolution",num_kernels=64,kernel_h=3,kernel_w=3,
convolution_params:{
    'stride':1
  })
</pre>
### To Add Relu Layer
<pre>model.add("relu")</pre>

### To Add Pooling Layer
<pre>model.add("pooling",pooling_params={
  "pooling_height":2,
  "pooling_width":2,
  "pooling_stride_height":2,
  'pooling_stide_width':2
  })
</pre>
### To Add a Flatten Layer
<pre>model.add("flatten")</pre>

### To Add Affine Layer
<pre>model.add("affine",affine_out=128)</pre>

### To Add Softmax Loss Layer
<pre>model.add("softmax")</pre>

### To Add SVM Loss Layer
<pre>model.add("svm")</pre>

### To Add MSE Loss Layer
<pre>model.add("mse")</pre>

### To Save Model
<pre>model.save("model.json")</pre>

### To Load Model
<pre>model = network.load("model.json")</pre>
