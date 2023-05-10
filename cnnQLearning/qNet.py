import torch
import torch.nn as nn
import torch.nn.functional as F

class qNet(nn.Module):
    def __init__(self):
        super(qNet, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.maxPool = torch.nn.MaxPool2d(2,2)
        self.layer2 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)
        self.layer3 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3)
        self.layer4 = torch.nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3)
        self.batchNorm32 = torch.nn.BatchNorm2d(32)
        self.batchNorm1 = torch.nn.BatchNorm2d(1)
        self.batchNorm10 = torch.nn.BatchNorm2d(10)
        self.relu = torch.torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(84, 20)
        self.linear2 = torch.nn.Linear(20, 6)
        self.batchNorm1D120 = torch.nn.BatchNorm1d(20)
        self.batchNorm1D6 = torch.nn.BatchNorm1d(6)



    def forward(self, nn_input):
        output1 = self.layer1(nn_input) # 550x350x3 to 548x348x32
        output1 = self.batchNorm32(output1) # 548x348x32
        output1 = self.relu(output1) # 548x348x32
        output1 = self.maxPool(output1) # 274x174x32

        output2 = self.maxPool(self.relu(self.batchNorm1(self.layer2(output1)))) # 136x66x128

        output3 = self.maxPool(self.relu(self.batchNorm32(self.layer3(output2)))) # 67x32x32

        output4 = self.maxPool(self.relu(self.batchNorm10(self.layer4(output3)))) # 32x15x10

        output5 = output2.view(-1, 84)

        output6 = self.linear2(self.relu(self.linear1(output5)))


        return output6



