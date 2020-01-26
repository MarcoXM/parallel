import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SingleModel(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(SingleModel,self).__init__()

        self.fc1 = nn.Linear(input_size,hidden1)
        self.fc2 = nn.Linear(hidden1,hidden2)
        self.fc3 = nn.Linear(hidden2,output_size)


    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class MultiModel(nn.Module):
    def __init__(self,input_size,hidden1,hidden2,output_size):
        super(MultiModel,self).__init__()

        self.fc1 = nn.Linear(input_size,hidden1)
        self.fc2 = nn.Linear(hidden1,hidden2)
        self.fc3 = nn.Linear(hidden2,output_size)


    def forward(self,inputs):
        x = torch.sigmoid(self.fc1(inputs))
        x = torch.sigmoid(self.fc2(x))
        outputs = torch.sigmoid(self.fc3(x))

        print("\tIn Model: input size", inputs.size(),
              "output size", outputs.size())
        return outputs


