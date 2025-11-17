import numpy as np

class LinearRegression:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights to zero
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias  # Linear prediction
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # Gradient wrt weights
            db = (1 / n_samples) * np.sum(y_pred - y)         # Gradient wrt bias
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
if __name__ == "__main__":
    # Input data (X) and target labels (y)
    X = np.array([[1], [2], [3], [4], [5]])  # Single feature
    y = np.array([1, 2, 3, 4, 5])            # Targets
    
    # Create and train the model
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Predict values
    predictions = model.predict(X)
    print("Predictions:", predictions)
