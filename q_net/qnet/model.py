import torch
from torch import nn


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
    