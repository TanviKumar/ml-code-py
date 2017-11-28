import mxnet as mx
from mxnet import nd
mx.random.seed(1)


x = nd.ones((3,4))
# print(x)

# Shape gives dimensions of the array
y = nd.random_normal(0,1, shape=(3,4))
# print(y.shape)
# print(y)

# print(x+y)

# print(x*y)
# print(nd.exp(y))
# print(nd.dot(x, y.T))
nd.elemwise_add(x, y, out=y)
# print(y)

# Splicing has the usual rules
# Broadcasting : expands one array to fit the other.
# nd.arange(x) -> goes from 0- n-1 values

a = x.asnumpy()
# print(a)
y = nd.array(a)
# print(y)


probabilities = nd.arange(6)
print(probabilities)
nd.sample_multinomial(probabilities)


