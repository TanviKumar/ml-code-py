

import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

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
        label_one_hot = nd.one_hot(label, 10)
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

# The following function was taken from the code of Geet Sharma on https://www.researchgate.net/post/How_to_create_MNIST_type_database_from_images
def imageprepare(argv):
    """
    This function returns the pixel values.
    The input is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    newImage.convert('1')
    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva

xBase=[imageprepare('./imgs/three.jpg')]#file path here
xBase = np.array(xBase)
xFinal = mx.nd.array(xBase)

newArr=[[0 for d in range(28)] for y in range(28)]
k = 0
for i in range(28):
    for j in range(28):
        newArr[i][j]=xBase[0][k]
        k=k+1

plt.imshow(newArr, interpolation='nearest')
plt.show()

pred=model_predict(net,xFinal)
print('model predictions are:', pred)