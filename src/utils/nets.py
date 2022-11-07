from torch import nn
import torch.nn.functional as F


class LinearFashionMNIST(nn.Module):
  def __init__(self):
    super(LinearFashionMNIST, self).__init__()

    self.flatten = nn.Flatten()
  
    self.linear_stack = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.Linear(128, 8)
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_stack(x)
    return logits


# TODO: add L2 regularization for CIFAR10

class CIFAR10Cnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.norm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,16,5)
        self.norm2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16,32,5)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,64,5)
        self.norm4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64,128,5)
        self.norm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,256,5)
        self.norm6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pool(out)
        
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = self.pool(out)
        
        out = torch.flatten(out, 1)
        
        out = self.fc1(out)
        return out
    
