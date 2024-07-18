import tensorflow as tf
import sonnet as snt

class SimpleNeuralNetwork(snt.Module):
	def __init__(self, output_size):
		super(SimpleNeuralNetwork, self).__init__()
		self._linear = snt.Linear(output_size=output_size)
	
	def __call__(self, x):
		return self._linear(x)

model = SimpleNeuralNetwork(output_size=10)
input_data = tf.random.normal([1,5])
output_data = model(input_data)
print(output_data)

