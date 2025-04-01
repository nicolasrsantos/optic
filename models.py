import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GATv2Conv, Linear, GraphConv

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, args):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        if args.linear == 1:
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, args):
        if args.weighted == 0:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)

        if args.linear == 1:
            x = F.relu(x)
            return self.classifier(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, args):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        if args.linear == 1:
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, args):
        if args.weighted == 0:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)

        if args.linear == 1:
            x = F.relu(x)
            return self.classifier(x)
        return x
    

class WeightedGraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, args):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = GraphConv(hidden_channels, out_channels, aggr='mean')
        if args.linear == 1:
            self.conv2 = GraphConv(hidden_channels, hidden_channels)
            self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, args):
        if args.weighted == 0:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)

        if args.linear == 1:
            x = F.relu(x)
            return self.classifier(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads)
        self.classifier = Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, edge_weight):
        h = self.conv1(x, edge_index)
        h = F.leaky_relu(h)
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h)

        return self.classifier(h), h
