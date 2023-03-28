#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class Model1(nn.Module):
  def __init__(self):
    super().__init__()
    #Convolutions
    self.conv1 = nn.Conv2d(3,6,3,bias = False)
    self.conv2 = nn.Conv2d(6,12,3, bias = False)
    self.conv3 = nn.Conv2d(12,16,3, bias = False)
    self.conv4 = nn.Conv2d(16,20,3, bias = False)

    #MLP Layer
    self.mlp = ops.MLP(2048, [512 for i in range(4)])

    #dropout
    self.dropout = nn.Dropout(0.5)

    #Recurrent Layers
    self.gru = nn.GRUCell(11520, 2048)

    #Linear Layers
    self.linear1 = nn.Linear(512, 256)
    self.linear2 = nn.Linear(256, 128)
    self.linear3 = nn.Linear(128,10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.dropout(x)
    
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    #x = self.dropout(x)
    x = torch.flatten(x,1)
    x = self.gru(x)
    x = F.relu(x)
    
    x = self.mlp(x)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)




    return x