import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class PredictiveCodingModel:
    def __init__(self, layer_sizes=[784, 256, 64, 10], lr_state=0.05, lr_weight=0.002):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.lr_state = lr_state
        self.lr_weight = lr_weight
        
        self.weights = [np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / (layer_sizes[i-1] + layer_sizes[i])) 
                        for i in range(1, self.num_layers)]
        self.biases = [np.zeros(layer_sizes[i-1]) for i in range(1, self.num_layers)]
        self.states = [np.zeros(size) for size in self.layer_sizes]
        self.errors = [np.zeros(size) for size in self.layer_sizes[:-1]]

    def reset_states(self):
        for i in range(1, len(self.states)):
            self.states[i].fill(0)
        for i in range(len(self.errors)):
            self.errors[i].fill(0)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self):
        self.predictions = []
        for i in range(1, self.num_layers):
            state_activity = self.relu(self.states[i])
            pred = np.dot(self.weights[i-1], state_activity) + self.biases[i-1]
            self.predictions.append(pred)
    
    def compute_errors(self):
        for i in range(self.num_layers - 1):
            self.errors[i] = self.states[i] - self.predictions[i]
            # --- FIX 1: Clip the errors to prevent them from exploding ---
            self.errors[i] = np.clip(self.errors[i], -1.0, 1.0)
    
    def update_states(self, clamp_output=False):
        for i in range(1, self.num_layers - 1):
            bottom_up_signal = np.dot(self.weights[i-1].T, self.errors[i-1])
            top_down_signal = self.errors[i]
            d_state = bottom_up_signal - top_down_signal
            self.states[i] += self.lr_state * d_state
            self.states[i] = self.relu(self.states[i])

        if not clamp_output:
            self.states[-1] += self.lr_state * np.dot(self.weights[-1].T, self.errors[-1])
            self.states[-1] = self.relu(self.states[-1])

    def update_weights(self):
        for i in range(self.num_layers-1):
            weight_update = self.lr_weight * np.outer(self.errors[i], self.relu(self.states[i+1]))
            bias_update = self.lr_weight * self.errors[i]
            
            self.weights[i] += weight_update
            self.biases[i] += bias_update
            # --- FIX 2: Re-introduce weight decay for regularization and stability ---
            self.weights[i] -= 0.0001 * self.weights[i]
    
    def train(self, data, target, num_iterations=50):
        self.reset_states()
        self.states[0] = data
        
        one_hot_target = np.zeros(self.layer_sizes[-1])
        one_hot_target[target] = 1
        self.states[-1] = one_hot_target
        
        for _ in range(num_iterations):
            self.forward()
            self.compute_errors()
            self.update_states(clamp_output=True)

        self.update_weights()
    
    def predict(self, input_data, num_iterations=50):
        self.reset_states()
        self.states[0] = input_data
        
        for _ in range(num_iterations):
            self.forward()
            self.compute_errors()
            self.update_states(clamp_output=False)
            
        output = self.states[-1]
        if np.sum(output) == 0: return np.random.randint(0, 10)
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / np.sum(exp_output)
        return np.argmax(probabilities)

# Data loading pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: torch.flatten(x))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model with slightly tuned learning rates for stability
model = PredictiveCodingModel(layer_sizes=[784, 256, 64, 10], lr_state=0.05, lr_weight=0.002)

# Training loop
num_epochs = 3
max_samples_per_epoch = 10000

print("Starting training...")
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        if i >= max_samples_per_epoch:
            break
        images_np = images.numpy().squeeze()
        labels_np = labels.numpy().item()
        
        model.train(images_np, target=labels_np)
        
        pred = model.predict(images_np)
        if pred == labels_np:
            correct += 1
        total += 1
        
        if (total % 1000 == 0) and total > 0:
            print(f"Epoch {epoch+1}, Sample {total}/{max_samples_per_epoch}, Running Training Accuracy: {100 * correct / total:.2f}%")
            
    print(f"--- Epoch {epoch+1} Final Training Accuracy: {100 * correct / total:.2f}% ---")

# Test loop
print("\nStarting testing...")
correct = 0
total = 0
max_test_samples = 2000

for i, (images, labels) in enumerate(test_loader):
    if i >= max_test_samples:
        break
    images_np = images.numpy().squeeze()
    labels_np = labels.numpy().item()
    
    pred = model.predict(images_np)
    if pred == labels_np:
        correct += 1
    total += 1

print(f"Final Test Accuracy on {total} samples: {100 * correct / total:.2f}%")