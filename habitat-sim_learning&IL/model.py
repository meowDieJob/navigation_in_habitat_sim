import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=(3,3),
            stride=1,
            padding=2)
        self.features1 = nn.Conv2d(
            in_channels=4,
            out_channels=3,
            kernel_size=(3,3),
            stride=1,
            padding=2)
        self.fc1 = nn.Linear(
            in_features=399384,
            out_features=1024,
        )
        self.fc2 = nn.Linear(
            in_features = 1024,
            out_features= 256
        )
        self.fc3 = nn.Linear(
            in_features=256,
            out_features=4
        )
    def forward(self,in0,in1):
        in0 = self.features(in0) # N x 3 x 256 x 256
        in1 = self.features1(in1)
        input = torch.cat((in0,in1),1) # N x 1024 x 8 x 8 1x 399384
        x = torch.flatten(input,start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x