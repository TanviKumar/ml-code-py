import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

# This takes a while.
ctx = mx.cpu()
mx.random.seed(1)            