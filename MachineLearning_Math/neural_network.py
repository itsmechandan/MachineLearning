import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)  # Weight matrix for input to hidden
        self.b1 = np.zeros((1, hidden_size))                # Bias for hidden layer
        self.W2 = np.random.randn(hidden_size, output_size) # Weight matrix for hidden to output
        self.b2 = np.zeros((1, output_size))                # Bias for output layer
        self.learning_rate = learning_rate

    # Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Derivative of sigmoid for backpropagation
    def sigmoid_derivative(self, z):
        return z * (1 - z)

    # Forward propagation
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1  # Input to hidden layer
        self.A1 = self.sigmoid(self.Z1)         # Activation for hidden layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Input to output layer
        self.A2 = self.sigmoid(self.Z2)         # Activation for output layer (final prediction)
        return self.A2

    # Backward propagation
    def backward(self, X, y, output):
        m = X.shape[0]  # Number of training examples

        # Compute the gradient of the loss with respect to the output (A2)
        dZ2 = output - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Backpropagate the error to the hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases using gradient descent
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # Training the network with given data
    def train(self, X, y, epochs=1000):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
    
    # Predict function for new data
    def predict(self, X):
        output = self.forward(X)
        return np.round(output)

# Example usage
if __name__ == "__main__":
    # Input data (X) and output labels (y)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR input
    y = np.array([[0], [1], [1], [0]])              # XOR output

    # Define and train the neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1)
    nn.train(X, y, epochs=10000)

    # Make predictions
    predictions = nn.predict(X)
    print("Predictions:\n", predictions)