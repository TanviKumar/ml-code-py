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

# Returns softmax
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

# Returns softmax cross entropy loss.
def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)                                  

# Defining the CNN.
def net(X, debug=False):
    #  Define the computation of the first convolutional layer
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3,3), num_filter=num_filter_conv_layer1)
    h1_activation = relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))

    #  Define the computation of the second convolutional layer
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(5,5), num_filter=num_filter_conv_layer2)
    h2_activation = relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))

    #  Flattening h2 so that we can feed it into a fully-connected layer
    h2 = nd.flatten(h2)
    # Adjust shape of W3 based on the shape of h2 after flattening
    if debug:
        print("Flat h2 shape: %s" % (np.array(h2.shape)))

    #  Define the computation of the third (fully-connected) layer
    h3_linear = nd.dot(h2, W3) + b3
    h3 = relu(h3_linear)
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))

    #  Define the computation of the output layer
    yhat_linear = nd.dot(h3, W4) + b4
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))

    return yhat_linear

# For debugging because shapes are confusing
# for i, (data, label) in enumerate(train_data):
#     data = data.as_in_context(ctx)
#     output = net(data, debug=True)
#     break

# Function for SGD for parameters
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

# Function to return accuracy
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

# Defining some variables for training the model
epochs = 5
learning_rate = .01
smoothing_constant = .01

# Traning loop
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, num_outputs)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        # Keeping moving average fo the loss
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))