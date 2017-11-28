import mxnet as mx
from mxnet import nd, autograd, gluon

# Setting context
data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000

# The actual Y thr real function.
def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

# This makes a list of lots of examples or basically a large sample of inputs.
# Noise makes it slightly more realistic
# Y is the actual value corresponding to each and every given X
X = nd.random_normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise

#This gets small parts of the training set that we have where X is the data and Y is the labels.
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                      batch_size=batch_size, shuffle=True)

# Dense model is suitable here because every input is matched to everything in the next layer.
# That makes it a dense neural network, one could say.
# 1 is the output dimension and in_units specifies the input dimension
# Our net will now have our parameters too!
# net.weight and net.bias 
net = gluon.nn.Dense(1, in_units=2)
# print(net.collect_params())

# Initialising the parameters. Ensures that the model can actually be called.
# The actual initialization is deferred until we make a first forward pass.
# Thus another step is essential before we can continue. That is that we must pass
# some data through our neural net so the actual initialisation of the parameters can take place.
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
example_data = nd.array([[4,7]])
net(example_data)

# Defining the loss function
square_loss = gluon.loss.L2Loss()

# Defining the optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0001})

# The actual training. Similar to how we built from scratch
epochs = 10
loss_sequence = []
num_batches = num_examples / batch_size

for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
    loss_sequence.append(cumulative_loss)

import matplotlib.pyplot as plt

plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)

plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)

plt.show()
