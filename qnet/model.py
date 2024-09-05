import torch
from torch import nn
import torch
import torch.nn as nn

class SClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_prob=0.3, activation_fn=nn.ReLU, use_residual=True):
        super(SClassifier, self).__init__()
        self.use_residual = use_residual
        self.fc_layers = self._create_fc_layers(input_size, hidden_sizes, dropout_prob, activation_fn)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation_fn = activation_fn()
        self.softmax = nn.Softmax(dim=1)

    def _create_fc_layers(self, input_size, hidden_sizes, dropout_prob, activation_fn):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  
            layers.append(activation_fn())
            layers.append(nn.Dropout(p=dropout_prob))
            prev_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.fc_layers(x)
        if self.use_residual and residual.shape == x.shape:
            x += residual 
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
    
class v1Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_prob=0.3):
        super(v1Classifier, self).__init__()
        self.fc_layers = self._create_fc_layers(input_size, hidden_sizes, dropout_prob)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.softmax = nn.Softmax(dim=1)

    def _create_fc_layers(self, input_size, hidden_sizes, dropout_prob):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # BatchNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))  # Dropout
            prev_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
    
def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
            
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_units, num_classes, dropout_rate):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_units)  # Input dimension is the embedding size (1536)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax across the class dimension
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        outputs = self.softmax(x)
        return outputs
    