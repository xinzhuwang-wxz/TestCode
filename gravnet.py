"""
@author     : wangxinzhu@sjtu.edu.cn
@date       : 2023-12-31 14:09:00
@brief      : gravnet
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from torch_geometric.nn import GravNetConv
import torch_cluster
try:
    from torch_cluster import knn
except ImportError:
    knn = None
class GravNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GravNetBlock, self).__init__()
        #self.global_exchange = nn.AdaptiveMaxPool2d((1, 1))
        self.dense1 = nn.Linear(in_channels , out_channels)
        self.dense2 = nn.Linear(out_channels, out_channels)
        self.dense3 = nn.Linear(out_channels, out_channels)
        self.message_passing1 = GravNetConv(out_channels, out_channels, space_dimensions=3, propagate_dimensions=1, k=3)
        self.message_passing2 = GravNetConv(out_channels, out_channels, space_dimensions=3, propagate_dimensions=1, k=3)
        self.message_passing3 = GravNetConv(out_channels, out_channels//2, space_dimensions=3, propagate_dimensions=1, k=3)
        self.message_passing4 = GravNetConv(out_channels//2, out_channels//2, space_dimensions=3, propagate_dimensions=1, k=3)
        self.message_passing5 = GravNetConv(out_channels//2, out_channels//4, space_dimensions=3, propagate_dimensions=1, k=3)
        self.message_passing6 = GravNetConv(out_channels//4, out_channels//4, space_dimensions=3, propagate_dimensions=1, k=3)
        self.dense4 = nn.Linear(out_channels//4, out_channels)
        self.dense5 = nn.Linear(out_channels, out_channels)
        self.dense6 = nn.Linear(out_channels, out_channels)
    
    def forward(self, inputs):
        
        #x = self.global_exchange(inputs.float())
        x = inputs.view(inputs.size(0), -1)
        
        x = self.dense1(x)
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
      
        x = self.message_passing1(x)
        x = self.message_passing2(x)
        x = self.message_passing3(x)
        x = self.message_passing4(x)
        x = self.message_passing5(x)
        x = self.message_passing6(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense4(x))
        x = F.relu(self.dense5(x))
        x = F.relu(self.dense6(x))
        return x
class GNNModel(torch.nn.Module):
    def __init__(self, classes):
        super(GNNModel, self).__init__()
        self.gravnet_block1 = GravNetBlock(in_channels=12960, out_channels=64)
        self.gravnet_block2 = GravNetBlock(in_channels=64, out_channels=64)
        self.gravnet_block3 = GravNetBlock(in_channels=64, out_channels=64)
        self.fc1 = nn.Linear(64 * 3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, classes)
        self.output = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_hits):
        x1 = self.gravnet_block1(input_hits)
        x2 = self.gravnet_block2(x1)
        x3 = self.gravnet_block3(x2)
        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.bn(x)
        output = F.relu(self.fc4(x))
        #output = self.sigmoid(self.output(x))
        return output
   
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = GNNModel(classes=2).to(device) 
    summary(t, (40, 18, 18))
