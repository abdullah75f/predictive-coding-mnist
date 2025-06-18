import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class PredictiveCodingModel:
    def __init__(self, layer_sizes=[784, 256, 64, 10], lr_state=0.1, lr_weight=0.01):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.lr_state = lr_state
        self.lr_weight = lr_weight
        self.states = [np.zeros(size) for size in layer_sizes]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.01 
                        for i in range(1, self.num_layers)]
        self.lateral_weights = [np.random.randn(size, size) * 0.01 
                                for size in layer_sizes[1:-1]]
        self.errors = [np.zeros(size) for size in layer_sizes[:-1]]
    
    def forward(self):
        self.predictions = []
        for i in range(1, self.num_layers):
            pred = np.dot(self.weights[i-1], self.states[i])
            self.predictions.append(pred)
        for i in range(1, self.num_layers-1):
            lateral_effect = np.dot(self.lateral_weights[i-1], self.states[i])
            self.states[i] += self.lr_state * lateral_effect
    
    def compute_errors(self):
        for i in range(self.num_layers-1):
            self.errors[i] = self.states[i] - self.predictions[i]
    
    def update_states(self):
        for i in range(1, self.num_layers-1):
            error_feedback = np.dot(self.weights[i].T, self.errors[i])
            self.states[i] += self.lr_state * (self.errors[i-1] - error_feedback)
        self.states[-1] += self.lr_state * np.dot(self.weights[-1].T, self.errors[-1])
    
    def update_weights(self):
        for i in range(self.num_layers-1):
            self.weights[i] += self.lr_weight * np.outer(self.errors[i], self.states[i+1])
        for i in range(len(self.lateral_weights)):
            self.lateral_weights[i] += self.lr_weight * np.outer(self.states[i+1], self.states[i+1])
    
    def train(self, data, num_iterations=20):  # Increased iterations
        for _ in range(num_iterations):
            self.forward()
            self.compute_errors()
            self.update_states()
            self.update_weights()
    
    def predict(self, input_data):
        self.states[0] = input_data
        self.forward()
        return np.argmax(self.states[-1])

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Initialize model
model = PredictiveCodingModel()

# Training loop
for epoch in range(1):
    for images, labels in train_loader:
        images = images.view(-1).numpy()
        model.states[0] = images
        model.train(images)
        pred = model.predict(images)
        print(f"Predicted: {pred}, Actual: {labels.item()}")