from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
import torch


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim, num_head, out_head):
        super(GAT, self).__init__()
        self.in_dim = in_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.out_dim = out_dim
        
        self.in_head = num_head
        self.out_head = out_head

        self.conv_in = GATConv(self.in_dim, self.hid_dim1, heads=self.in_head) #dropout=0.6
        self.conv_mid = GATConv(self.hid_dim1*self.in_head, self.hid_dim2, heads=self.in_head)
        self.conv_out = GATConv(self.hid_dim2*self.in_head, self.out_dim, concat=False, heads=self.out_head)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
        
    def forward(self, data):
        x, edge_index = data.node_attrs, data.edge_index
        

        x = self.conv_in(x, edge_index)
        x = F.elu(x)
        
        x = self.conv_mid(x, edge_index)
        x = F.elu(x)       

        x = self.conv_out(x, edge_index)
        x = F.elu(x)

        x = self.fc1(x)
        x = F.elu(x)

        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x
