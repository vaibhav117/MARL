import numpy as np
import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package

device = torch.device("cuda")

"""
This class has the basic DQN declaration
We use a 3 fairly simple layer conv-net 
followed by 4 fully connected layers.
"""

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        torch_shape = (shape[2] , shape[0] , shape[1])
        o = self.conv(torch.zeros(1, *torch_shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.permute(0,3,1,2)
        conv_out = self.conv(x)
        return self.fc(conv_out)