from neuralnetwork import neuralnetwork
import numpy as np

layer_sizes = (784, 16, 16, 10)

net = neuralnetwork(layer_sizes)
inputs = np.ones((layer_sizes[0], 1))
prediction = net.predict(inputs)
print(prediction)