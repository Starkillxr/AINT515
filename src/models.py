#imports
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class Model1(nn.Module):
  def __init__(self):
    super().__init__()
    #Convolutions
    self.conv1 = nn.Conv2d(3,6,5)
    self.conv2 = nn.Conv2d(6,16,5)
  
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
  
#Solved issues with Model 2 using the thread at https://discuss.pytorch.org/t/valueerror-expected-input-batch-size-324-to-match-target-batch-size-4/24498
class Model2(nn.Module):
  def __init__(self):
    super().__init__()
    #Convolutions
    self.conv1 = nn.Conv2d(3,6,5)
    self.conv2 = nn.Conv2d(6,16,5)
    self.conv3 = nn.Conv2d(16,16,3)
  
    #dropout
    self.dropout = nn.Dropout(0.5)
    
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.pool1 = nn.MaxPool2d(3)
    self.pool3 = nn.MaxPool2d(5)


    #Linear Layers
    self.linear1 = nn.Linear(16, 120)
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
    
    x = self.conv3(x)
    x = F.relu(x)
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    x = self.linear1(x)
    x = F.relu(x)
    
    x = self.linear2(x)
    x = F.relu(x)
    
    x = self.linear3(x)

    return x