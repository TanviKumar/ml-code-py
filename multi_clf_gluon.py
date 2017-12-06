# Using MNIST

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

# Defining some basic values.
num_inputs = 784
num_outputs = 10
num_examples = 60000

batch_size = 64
# Loading the data iterator
train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

# Defining the neural net. Here it is dense as each node in one layer in connected to all in the other.
net = gluon.nn.Dense(num_outputs)

# Parameter initialisation
# The parameters get initialised during the first call to the forward method.
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

# Optimiser usung gluon.
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# Evalution metric. Returns the accuracy.
def evaluate_accuracy(data_iterator, net):
	#Accuracy using metric package
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        # The values given by our net stored in output
        output = net(data)
        # The prediction will naturally be the one which has the maximum value in output
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

# Loss function.
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()