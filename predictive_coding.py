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
             # Initialize states
             self.states = [np.zeros(size) for size in layer_sizes]
             # Initialize forward weights (predict lower layer)
             self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.01 
                             for i in range(1, self.num_layers)]
             # Initialize lateral weights (within each hidden layer)
             self.lateral_weights = [np.random.randn(size, size) * 0.01 
                                     for size in layer_sizes[1:-1]]
             # Initialize errors
             self.errors = [np.zeros(size) for size in layer_sizes[:-1]]