#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class Model1(nn.Module):
  def __init__(self):
    super().__init__()
    #Convolutions
    self.conv1 = nn.Conv2d(3,6,5)
    self.conv2 = nn.Conv2d(6,16,5)

    #MLP Layer
    self.mlp = ops.MLP(1800, [900 for i in range(9)])

    #dropout
    self.dropout = nn.Dropout(0.5)
    
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.pool1 = nn.MaxPool2d(3)
    self.pool3 = nn.MaxPool2d(5)


    #Linear Layers
    self.linear1 = nn.Linear(16*5*5, 120)
    self.linear2 = nn.Linear(120, 84)
    self.linear3 = nn.Linear(84,10)

  def forward(self, x):
    #Convolutions
    x = x
    
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool(x)

    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool(x)
    
    x = x.reshape(-1, 16*5*5)
    x = self.linear1(x)
    x = F.relu(x)
    
    x = self.linear2(x)
    x = F.relu(x)
    
    x = self.linear3(x)

    return x