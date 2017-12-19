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