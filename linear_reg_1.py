import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import matplotlib.pyplot as plt

mx.random.seed(1)


data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

# This makes a list of lots of examples or basically a large sample of inputs.
# Noise makes it slightly more realistic
# Y is the actual value corresponding to each and every given X
X = nd.random_normal(shape = (num_examples, num_inputs), ctx = data_ctx)
noise = .1 * nd.random_normal(shape = (num_examples,), ctx = data_ctx)
y = real_fn(X) + noise

# plt.scatter(X[:, 1].asnumpy(),y.asnumpy())
# plt.show()

#This gets small parts of the training set that we have where X is the data and Y is the labels.
batch_size = 4;
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X,y), batch_size = batch_size, shuffle = True)

# for i, (data, label) in enumerate(train_data):
# 	print(data, label)
# 	break

# We are trying to estimate the value of w for each part of the input and also the value of b which is the constant.
# w is known as the weight or the constant X is multiplied with. All multiplicative parameters are weights.
# We are giving all the parameters an initial random value which we will try to manipulate over training
# so it may lead back to the so called correct value.
w = nd.random_normal(shape = (num_inputs, num_outputs), ctx = model_ctx)
b = nd.random_normal(shape = (num_outputs), ctx = model_ctx)
# We are saving all the gradients wrt all parameters
params = [w,b]

for param in params:
    param.attach_grad()

# Now we define the neural net. Taking a linear model, we multiply the input or X with the weights and add b.
def net(X):
	return mx.nd.dot(X, w) + b

# We need to keep reducing the loss. But what is the loss? Here lets take the square of the differnce from y 
# Remember y is the expected or correct value.
# Thus, the loss function
def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)

# Stochastic gradient descent. Here we are trying to make the parameters change themselves such that they move towards 
# the actual value or the parameters in y. Currently they are random values.
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

# Epochs refers to number of times you go throught the entire sample itself
# Learning rate is the rate by which the parameters move itself towards the correct value.
# It should not be too large as it may not value the past much. It must also not be too small as it won't be correct
# even after seeing many samples.
epochs = 15
learning_rate = .0001
num_batches = num_examples/batch_size
losses = []

# The training loop
# We are going to see how the loss reduces.
for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
        	# The output is the experimental y value.
            output = net(data)
            # Compares with actual y values and returns the mean square loss.
            loss = square_loss(output, label)
        # Calculating the gradients.
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()
    # Prints the mean loss in a giving traversal through whole set.
    print("Epoch %s, batch %s. Mean loss: %s" % (e, i, cumulative_loss/num_batches))
    losses.append(cumulative_loss/num_batches)

# Check out the value of params and how close it is to the actual parameters of y.
#print(params)

# Now lets plot the loss after testing.

def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()

    plt.show()


plot(losses, X)

print(w)
print(b)
