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
print(image.shape, label)

# matplotlib expects either (height, width) or (height, width, channel) where channel has RGB values.
# Hence we broadcast single channel to three.
# Print the shape to get a clearer idea.
im = mx.nd.tile(image, (1,1,3))
print(im.shape)

# You will see a clear five when you implement plt.show().
plt.imshow(im.asnumpy())
plt.show()