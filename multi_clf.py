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

# Just to get an idea about what softmax function is doing to a bunch of random values.    
# sample_y_linear = nd.random_normal(shape=(2,10))
# sample_yhat = softmax(sample_y_linear)
# print(sample_yhat)

# Defining our model or neural network
def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat

# Our loss function. 
# This only cares about the prediction made by our net for the correct label, none of the others.
def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat), axis=0, exclude=True)

# The Stochastic Gradient Descent optimiser for our net.
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

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
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, num_outputs)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

# The predictor. Returns prediction when we use our net.
def model_predict(net,data):
    output = net(data)
    return nd.argmax(output, axis=1)

samples = 10
# Sampling 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mnist_test, samples, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    print(data.shape)
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