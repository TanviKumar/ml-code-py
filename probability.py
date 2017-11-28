# Deals with only one random variable

import mxnet as mx
from mxnet import nd
import matplotlib
from matplotlib import pyplot as plt

num = 3000

probabilities = nd.ones(6) / 6
rolls = nd.sample_multinomial(probabilities, shape=(num))

counts = nd.zeros((6,num))
totals = nd.zeros(6)

# Counting the number of trials at each step and the total number of rolls
for i, roll in enumerate(rolls):
	totals[ int(roll.asscalar())] += 1
	counts[:, i] = totals

# Generating the probability at each instant by creating an array of 1-n

x = nd.arange(num).reshape((1,num)) + 1
estimates = counts / x
# print(estimates[:, 0])
# print(estimates[:, 1])
# print(estimates[:, num - 1])

# Plotting all of the choices and their probability
plt.plot(estimates[0, :].asnumpy(), label="Estimated P(die=1)")
plt.plot(estimates[1, :].asnumpy(), label="Estimated P(die=2)")
plt.plot(estimates[2, :].asnumpy(), label="Estimated P(die=3)")
plt.plot(estimates[3, :].asnumpy(), label="Estimated P(die=4)")
plt.plot(estimates[4, :].asnumpy(), label="Estimated P(die=5)")
plt.plot(estimates[5, :].asnumpy(), label="Estimated P(die=6)")

# Plotting the actual theortical probabilty value
plt.axhline(y=0.16666, color='black', linestyle='dashed')
plt.legend()
plt.show()