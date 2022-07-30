# Net used in common situation

from turtle import forward
import torch
from torch import nn

# Building

class Lr(nn.Module):
    def __init__(self) -> None:
        super(Lr, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2),  # kernel_size
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),  # reshape the input to be linear
            nn.Linear(64*4*4, 64),  # in_features, out_features
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.model(x)  # capture the var x, return the "netted" x
        return x

if __name__ == "__main__":
    Lr = Lr()
    input = torch.ones((64, 3, 32, 32))  # size
    output = Lr(input)
    print(output.shape)
