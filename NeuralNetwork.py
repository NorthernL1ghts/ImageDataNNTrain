import numpy as np
import os
import cv2

# Define the NeuralNetwork class:
class NeuralNetwork:
    # Initialize the network:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        # Initialize the network:
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        # Initialize the weights for the input-to-hidden and hidden-to-output layers:
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)
        
    # Define the sigmoid activation function:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Define the derivative of the sigmoid activation function:
    def sigmoid_derivative(self, z):
        return np.exp(-z) / ((1 + np.exp(-z))**2)

    # Define the forward pass of the network:
    def forward(self, X):
        # Calculate the activations in the hidden layer:
        self.z2 = X.dot(self.W1)
        self.a2 = self.sigmoid(self.z2)
        
        # Calculate the activations in the output layer:
        self.z3 = self.a2.dot(self.W2)
        y_hat = self.sigmoid(self.z3)
        return y_hat
    
    # Define the cost function used to train the network:
    def cost_function(self, X, y):
        y_hat = self.forward(X)
        J = 0.5 * sum((y - y_hat)**2)
        return J
    
    # Define the derivative of the cost function with respect to the weights:
    def cost_function_derivative(self, X, y):
        y_hat = self.forward(X)
        
        # Calculate the error in the output layer:
        delta3 = np.multiply(-(y - y_hat), self.sigmoid_derivative(self.z3))
        
        # Calculate the gradient for the hidden-to-output weights:
        djdw2 = self.a2.T.dot(delta3)
        
        # Calculate the error in the hidden layer:
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(self.z2)
        
        # Calculate the gradient for the input-to-hidden weights:
        djdw1 = X.T.dot(delta2)
        return djdw1, djdw2
    
    # Train the network:
    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            djdw1, djdw2 = self.cost_function_derivative(X, y)
            self.W1 -= learning_rate * djdw1
            self.W2 -= learning_rate * djdw2
        
def read_images(path):
    X = []
    y = []
    for image_path in os.listdir(path):
        image = cv2.imread(os.path.join(path, image_path), 0)
        image = image.flatten() / 255
        X.append(image)
        label = int(image_path.split(".")[0])
        y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = read_images("images")
    nn = NeuralNetwork(input_layer_size, hidden_layer_size, output_layer_size)
    nn.train(X, y, epochs=1000, learning_rate=0.01)
    
    while True:
        file_path = input("Enter file path: ")
        if file_path == "exit":
            break
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = image.flatten()
        prediction = nn.predict(image)
        print(f"The car brand is: {prediction}")

       
