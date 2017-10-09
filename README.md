# Modular Neural Network Library
Neural Networks From Scratch. The Library was Trained & Tested on Gesture Recognition dataset created by <a href="https://github.com/ankitesh97">Ankitesh Gupta</a>

The images were Normalized using the <b>Mean Pixel Value</b> and the <b>Standard Deviation of the Pixel Value</b> before giving it to the model for <b>Training and Testing</b>

### Example Usage
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
<pre>model.add("Relu")</pre>

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

### To Add a Loss Layer
<pre>model.add("softmax")</pre>

### To Save Model
<pre>model.save("model.pkl")</pre>

### To Load Model
<pre>model = NN.load("model.pkl")</pre>

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
  i) Kernels : 64 \
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

<b>Training Accuracy is 100%</b>

<b>Validation Accuracy is 100%</b>

<b>Test Accuracy:  99.8%</b>
