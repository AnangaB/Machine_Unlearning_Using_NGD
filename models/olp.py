import torch.nn as nn
import torch
# One-Layer Perceptron (OLS) definition
class OLP(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_size):
        super(OLP, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, output_size)
    
    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x) 
        return x
