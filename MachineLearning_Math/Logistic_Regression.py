import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid function

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights to zero
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)  # Apply sigmoid function

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # Gradient wrt weights
            db = (1 / n_samples) * np.sum(y_pred - y)         # Gradient wrt bias

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)  # Return probabilities

    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        return [1 if prob >= 0.5 else 0 for prob in y_pred_proba]  # Convert probabilities to class labels

# Example usage
if __name__ == "__main__":
    # Input data (X) and binary target labels (y)
    X = np.array([[1], [2], [3], [4], [5]])  # Single feature
    y = np.array([0, 0, 1, 1, 1])            # Binary targets

    # Create and train the model
    model = LogisticRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Predict values
    predictions = model.predict(X)
    print("Predictions:", predictions)