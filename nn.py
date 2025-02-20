import numpy as np
from utils import mse

class NeuralNetwork():


  def __init__(self, input_size, hidden_size, output_size, activation = "relu"):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.random.randn(1, hidden_size)

        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.random.randn(1, output_size)


        if activation == "sigmoid":
            self.activation = self.sigmoid
            self.activation_deriv = self.sigmoid_deriv
        elif activation == "tanh":
            self.activation = self.tanh
            self.activation_deriv = self.tanh_deriv
        elif activation == "relu":
            self.activation = self.relu
            self.activation_deriv = self.relu_deriv
        else:
            raise ValueError("Invalid. Choose 'sigmoid', 'tanh', or 'relu'.")


  @staticmethod
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))

  @staticmethod
  def sigmoid_deriv(x):
      return x * (1 - x)

  @staticmethod
  def tanh(x):
      return np.tanh(x)

  @staticmethod
  def tanh_deriv(x):
      return 1 - np.tanh(x) ** 2

  @staticmethod
  def relu(x):
      return np.maximum(0, x)

  @staticmethod
  def relu_deriv(x):
      return np.where(x > 0, 1, 0)

  @staticmethod
  def mse(y, y_pred):
      return np.mean(np.square(y - y_pred))



  def forward(self, x):

          self.z1 = np.dot(x, self.weights1) + self.bias1
          self.a1 = self.activation(self.z1)

          self.z2 = np.dot(self.a1, self.weights2) + self.bias2
          self.a2 = self.activation(self.z2)

          return self.a2


  def backward(self, x, y, output, learning_rate):

        error1 = y - self.a2
        delta1 = error1 * self.activation_deriv(self.a2)

        self.weights2 += np.dot(self.a1.T, delta1) * learning_rate
        self.bias2 += np.sum(delta1, axis=0, keepdims=True) * learning_rate

        error2 = np.dot(delta1, self.weights2.T)
        delta2 = error2 * self.activation_deriv(self.a1)

        self.weights1 += np.dot(x.T, delta2) * learning_rate
        self.bias1 += np.sum(delta2, axis=0, keepdims=True) * learning_rate


  def train(self, x, y, epochs, learning_rate):
          for i in range(epochs):
              output = self.forward(x)
              self.backward(x, y, output, learning_rate)

              if (i + 1) % 1000 == 0:
                  loss = self.mse(y, output)
                  print(f"Epoch {i+1}/{epochs} - Loss: {loss}")


x = np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]])

y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(3, 4, 1, activation="sigmoid")
nn.train(x, y, 10000, 0.1)

yp = nn.forward(x)

print("\n", mse(y, yp))


"""
Epoch 1000/10000 - Loss: 0.05387997770131594
Epoch 2000/10000 - Loss: 0.005213199131301026
Epoch 3000/10000 - Loss: 0.0023247200541719657
Epoch 4000/10000 - Loss: 0.0014476258390274698
Epoch 5000/10000 - Loss: 0.0010367478108108917
Epoch 6000/10000 - Loss: 0.000801564969976938
Epoch 7000/10000 - Loss: 0.0006503452711209265
Epoch 8000/10000 - Loss: 0.0005454199319568983
Epoch 9000/10000 - Loss: 0.0004685946798566749
Epoch 10000/10000 - Loss: 0.00041004808430059557

MSE: 0.00040999654407552696

"""