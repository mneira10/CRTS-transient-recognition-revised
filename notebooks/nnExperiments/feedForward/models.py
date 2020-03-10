import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.n1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.n2 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.n1(x)
        out = self.fc1(out)
        out = self.n2(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


class GeneralFeedForward(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(NeuralNet, self).__init__()
        self.n1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.n2 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        s

    def forward(self, x):
        out = self.n1(x)
        out = self.fc1(out)
        out = self.n2(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out