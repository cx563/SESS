import torch.nn as nn
import torch.nn.functional as F
class GCN(nn.Module):

    def __init__(self, emb_dim=768, num_layers=2,gcn_dropout=0.1):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)


    def forward(self, adj, inputs):

        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.layers):
            Ax = adj.bmm(inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](inputs)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return inputs, mask