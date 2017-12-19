import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

# This takes a while.
ctx = mx.cpu()
mx.random.seed(1)

batch_size = 64
num_inputs = 784
num_outputs = 10

# Init DataLoader
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

# Init Parameters
weight_scale = .01
num_fc = 128
num_filter_conv_layer1 = 20
num_filter_conv_layer2 = 20

W1 = nd.random_normal(shape=(num_filter_conv_layer1
	, 1, 3,3), scale=weight_scale, ctx=ctx)
b1 = nd.random_normal(shape=num_filter_conv_layer1, scale=weight_scale, ctx=ctx)

W2 = nd.random_normal(shape=(num_filter_conv_layer2, num_filter_conv_layer1, 5, 5), scale=weight_scale, ctx=ctx)
b2 = nd.random_normal(shape=num_filter_conv_layer2, scale=weight_scale, ctx=ctx)

W3 = nd.random_normal(shape=(320, num_fc), scale=weight_scale, ctx=ctx)
b3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

W4 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
b4 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Attaching gradient to parameters
for param in params:
    param.attach_grad()
    
###### See how the convolution and pooling changes the shape.
# for data, _ in train_data:
#     data = data.as_in_context(ctx)
#     break
# conv = nd.Convolution(data=data, weight=W1, bias=b1, kernel=(3,3), num_filter=20)
# print(conv.shape)

# pool = nd.Pooling(data=conv, pool_type="max", kernel=(2,2), stride=(2,2))
# print(pool.shape) 

# Usual Rectified Linear Unit function
def relu(X):
    return nd.maximum(X,nd.zeros_like(X))                                  