# Multilayer Perceptron
# Using deep neural net to predict MNIST test data
# We have two hidden layers in this example.
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt

# Setting the context
data_ctx = mx.cpu()
model_ctx = mx.cpu()

# Assigning basic values
num_inputs = 784
num_outputs = 10
batch_size = 64
num_examples = 60000

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

train_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

#  Set some constants so it's easy to modify the network later
num_hidden = 256
weight_scale = .01

#  Allocate parameters for the first hidden layer
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=model_ctx)
b1 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

#  Allocate parameters for the second hidden layer
W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=model_ctx)
b2 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

#  Allocate parameters for the output layer
W3 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=model_ctx)
b3 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=model_ctx)

# These are all our parameters
params = [W1, b1, W2, b2, W3, b3]

# Assigning space for the gradient of all the parameters
for param in params:
    param.attach_grad()

# rectified linear unit
# Defining the activation function
def relu(X):
    return nd.maximum(X, nd.zeros_like(X))

# The softmax output
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition

# Loss function
def cross_entropy(yhat, y):
    return - nd.nansum(y * nd.log(yhat), axis=0, exclude=True)

# Our neural net
def net(X):
    #  Compute the first hidden layer
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)

    #  Compute the second hidden layer
    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)

    #  Compute the output layer.
    yhat_linear = nd.dot(h2, W3) + b3

    #  Computing the softmax of output layer for final output.
    yhat_linear = softmax(yhat_linear)
    return yhat_linear       

# Stochastic Gradient Descent Optimiser definition
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

# Function to calculate accuracy
# Evaluating accuracy, i.e, number of times our net gets the right answer over the total number of tries.
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0. 
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        # The values given by our net stored in output
        output = net(data)
        # The prediction will naturally be the one which has the maximum value in output.
        predictions = nd.argmax(output, axis=1)
        # Count only those which match the true label for numerator
        numerator += nd.sum(predictions == label)
        # Total number of checks
        denominator += data.shape[0]
    # Returning the accuracy of the net, or the probability of getting label right using our net.
    return (numerator / denominator).asscalar()

epochs = 10
learning_rate = .001

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

# The predictor. Returns prediction when we use our net.
def model_predict(net,data):
    output = net(data)
    return nd.argmax(output, axis=1)

samples = 10

# Sampling 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), samples, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    # print(data.shape)
    im = nd.transpose(data,(1,0,2,3))
    im = nd.reshape(im,(28,samples*28,1))
    imtiles = nd.tile(im, (1,1,3))
    # Seeing the predictions after the training is done
    plt.imshow(imtiles.asnumpy())
    plt.show()
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
    print('true labels :', label)
    break