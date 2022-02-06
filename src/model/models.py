import torch
import torch.nn.functional as F


class RegressorNet(torch.nn.Module):
    """Simple Multilayer Perceptron (MLP) for house price prediction"""

    def __init__(self, n_input, n_output=1):
        super(RegressorNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_input, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, n_output)

    def forward(self, x):
        """forward pass"""
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x