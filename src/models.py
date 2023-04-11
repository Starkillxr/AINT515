#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class Model1(nn.Module):
  def __init__(self):
    super().__init__()
    #Convolutions
    self.conv1 = nn.Conv2d(3,30,3,bias = False)
    self.conv2 = nn.Conv2d(30,300,3, bias = False)
    self.conv3 = nn.Conv2d(300,600,3, bias = False)
    self.conv4 = nn.Conv2d(600,1800,3, bias = False)

    #MLP Layer
    self.mlp = ops.MLP(1800, [900 for i in range(9)])

    #dropout
    self.dropout = nn.Dropout(0.5)
    
    self.pool = nn.MaxPool2d(3)
    self.pool1 = nn.MaxPool2d(3)
    self.pool2 = nn.MaxPool2d(3)
    self.avg = nn.AvgPool2d(5)


    #Linear Layers
    self.linear1 = nn.Linear(900, 300)
    self.linear2 = nn.Linear(300, 100)
    self.linear3 = nn.Linear(100,10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(F.relu(self.conv2(x)))
    x = self.dropout(x)
    
    x = F.relu(self.conv3(x))
    x = self.avg(F.relu(self.conv4(x)))
    #x = self.dropout(x)
    x = torch.flatten(x,1)
    x = self.mlp(x)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)




    return x