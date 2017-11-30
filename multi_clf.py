import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt
import numpy as np

data_ctx = mx.cpu()
model_ctx = mx.cpu()

# Transforming data into tuples
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

# Proof that the image is now a 3-tuple. The label is the actual number in the image.
image, label = mnist_train[0]
# print(image.shape, label)

# matplotlib expects either (height, width) or (height, width, channel) where channel has RGB values.
# Hence we broadcast single channel to three.
# Print the shape to get a clearer idea.
im = mx.nd.tile(image, (1,1,3))
# print(im.shape)

# You will see a clear five when you implement plt.show().
# plt.imshow(im.asnumpy())
# plt.show()

# Defining some basic values.
num_inputs = 784
num_outputs = 10
num_examples = 60000

batch_size = 64
# Loading the data iterator
train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

# Allocating model parameters
W = nd.random_normal(shape=(num_inputs, num_outputs),ctx=model_ctx)
b = nd.random_normal(shape=num_outputs,ctx=model_ctx)
# Attaching gradient to all the parameters
params = [W, b]
for param in params:
    param.attach_grad()

# Defining the softmax
def softmax(y_linear):
    # exponent of a negative value is always between 0 and 1. 
    exp = nd.exp(y_linear-nd.max(y_linear))
    # Finding the total sum of all the exponents 
    norms = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    # Retrning value of exponent by total sum such that all of them overall sum up to 1.
    return exp / norms