# Modular Neural Network Library
Neural Networks From Scratch. The Library was Trained & Tested on Gesture Recognition dataset created by <a href="https://github.com/ankitesh97">Ankitesh Gupta</a>

The images were Normalized using the <b>Mean Pixel Value</b> and the <b>Standard Deviation of the Pixel Value</b> before giving it to the model for <b>Training and Testing.</b> The code for normalizing the data is in preprocess.py

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
model  = NN(input_shape=(64,1,50,50),initialization="xavier2",
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
<pre>model = NN.load("model.json")</pre>

## Model Results On Gesture Recognition

The Model Consists of the Following Layers:

1) Zero Padding Layer \
  i) Height Padding : 2 \
 ii) Width Padding : 2

2) Convolution Layer \
  i) Kernels : 64 \
 ii) Kernel Height : 3 \
iii) Kernel Width  : 3 \
iv ) Stride        : 1

3) Pooling Layer \
  i) Pooling Height : 2 \
 ii) Pooling Width : 2 \
iii) Stride Height : 2 \
 iv) Stride Width  : 2

4) Relu Layer

5) Zero Padding Layer \
  i) Height Padding : 2 \
 ii) Width Padding : 2

6) Convolution Layer \
  i) Kernels : 128 \
 ii) Kernel Height : 3 \
iii) Kernel Width  : 3 \
iv ) Stride        : 1

7) Pooling Layer \
  i) Pooling Height : 2 \
 ii) Pooling Width : 2 \
iii) Stride Height : 2 \
 iv) Stride Width  : 2
8) Relu Layer

9) Flatten Layer

10) Affine Layer : 128 Neurons

11) Affine Layer : 64 Neurons

12) Affine Layer : 16 Neurons

13) Affine Layer : 5 Neurons (This is the Output Layer)

14) Softmax Layer

After <b>100 Epochs</b> The Model Performance is:

<b>Training Accuracy: 100%</b>

<b>Validation Accuracy: 100%</b>

<b>Test Accuracy:  99.8%</b>

It took about **1-1.5 hour** to train this model for 100 Epochs. \
Initial Weights of the Network were assigned using Xavier Initialization. \
The Model was trained using Mini-Batch Gradient Descent with Adam Optimizer. \
The Mini-Batch was sampled at random during training.
### Loss-Iteration Curve
![Loss-Iteration Curve for 100 Epochs](/Loss_Curve.png)

### Model File
The Model file can be downloaded from <a href="https://drive.google.com/open?id=0B6OWaNVUCQvaeWo3aTJoWlpOdWM">here</a>

Extract it in models folder &
Load the model file as follows:
<pre>model  = NN.load("model.json")</pre>
