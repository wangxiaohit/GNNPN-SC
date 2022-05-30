import torch
import torch.nn.functional as F
from torch.nn import Embedding, ModuleList
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Sigmoid
from torch_geometric.nn import GINConv, GCNConv
from torch_scatter import scatter


class NodeEncoder(torch.nn.Module):
    def __init__(self, hiddenChannels):
        super(NodeEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(9):
            self.embeddings.append(Embedding(100, hiddenChannels))  # 100->500

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i].long())
        return out


class EdgeEncoder(torch.nn.Module):
    def __init__(self, hiddenChannels):
        super(EdgeEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(9):
            self.embeddings.append(Embedding(100, hiddenChannels))  # 100->500

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        out = 0
        for i in range(edge_attr.size(1)):
            out += self.embeddings[i](edge_attr[:, i])
        return out


class Net(torch.nn.Module):
    def __init__(self, hiddenChannels, outChannels, embeddingChannels, numLayersGIN, numLayersGCN, isServices=True, dropout=0.0):
        super(Net, self).__init__()
        self.sigmoid = Sigmoid()

        self.numLayersGIN = numLayersGIN
        self.numLayersGCN = numLayersGCN
        self.dropout = dropout
        self.outChannels = outChannels
        self.reqAndServiceChannels = embeddingChannels
        self.qosNumber = 4  # 2->4
        self.constraintNumber = 2  # 2->4
        self.isService = isServices

        self.nodeEncoder = NodeEncoder(self.reqAndServiceChannels)
        self.serviceEncoder = NodeEncoder(self.reqAndServiceChannels)

        self.nodeConvs = ModuleList()
        self.nodeBatchNorms = ModuleList()
        first = True
        for _ in range(numLayersGIN):
            if first:
                nn = Sequential(
                    Linear(self.reqAndServiceChannels + self.constraintNumber * 3, 2 * hiddenChannels),
                    BatchNorm1d(2 * hiddenChannels),
                    ReLU(),
                    Linear(2 * hiddenChannels, hiddenChannels),
                )
                first = False
            else:
                nn = Sequential(
                    Linear(hiddenChannels, 2 * hiddenChannels),
                    BatchNorm1d(2 * hiddenChannels),
                    ReLU(),
                    Linear(2 * hiddenChannels, hiddenChannels),
                )
            self.nodeConvs.append(GINConv(nn, train_eps=True))
            self.nodeBatchNorms.append(BatchNorm1d(hiddenChannels))
        self.nodeLin = Linear(hiddenChannels, hiddenChannels)

        # self.serviceEdgeEncoders = ModuleList()
        self.serviceConvs = ModuleList()
        self.serviceBatchNorms = ModuleList()
        first = True
        for i in range(numLayersGCN):
            if first:
                self.serviceConvs.append(GCNConv(self.reqAndServiceChannels + self.qosNumber, 2 * hiddenChannels))
                first = False
            else:
                self.serviceConvs.append(GCNConv(2 * hiddenChannels, 2 * hiddenChannels))
            self.serviceBatchNorms.append(BatchNorm1d(2 * hiddenChannels))

        self.serviceLin = Linear(2 * hiddenChannels, hiddenChannels)

        self.noServicesLins = ModuleList()
        first = True
        for i in range(numLayersGCN):
            if first:
                self.noServicesLins.append(Linear(self.reqAndServiceChannels + self.qosNumber, 2 * hiddenChannels))
                first = False
            else:
                self.noServicesLins.append(Linear(2 * hiddenChannels, 2 * hiddenChannels))

    def reset_parameters(self):
        self.nodeEncoder.reset_parameters()
        self.serviceEncoder.reset_parameters()

        for conv, batch_norm in zip(self.nodeConvs, self.nodeBatchNorms):
            conv.reset_parameters()
            batch_norm.reset_parameters()
        self.nodeLin.reset_parameters()

        for conv, batch_norm in zip(self.serviceConvs, self.serviceBatchNorms):
            conv.reset_parameters()
            batch_norm.reset_parameters()
        self.serviceLin.reset_parameters()

    def forward(self, data):

        x = data.x.squeeze()
        x1 = x[:, 0].view(-1, 1).long()
        x2 = x[:, 1:]
        x1 = self.nodeEncoder(x1)
        x = torch.cat((x1, x2), -1)

        for i in range(self.numLayersGIN):
            x = self.nodeConvs[i](x, data.edge_index)
            x = self.nodeBatchNorms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x_service = data.x_service.squeeze()
        x1 = x_service[:, 0].view(-1, 1).long()
        x2 = x_service[:, 1:]
        x1 = self.serviceEncoder(x1)
        x_service = torch.cat((x1, x2), -1)

        if self.isService:
            for i in range(self.numLayersGCN):
                x_service = self.serviceConvs[i](x_service, data.edge_index_service, data.edge_attr_service)
                x_service = self.serviceBatchNorms[i](x_service)
                x_service = F.relu(x_service)
                x_service = F.dropout(x_service, self.dropout, training=self.training)
        else:
            for i in range(self.numLayersGCN):
                x_service = self.noServicesLins[i](x_service)
                x_service = self.serviceBatchNorms[i](x_service)
                x_service = F.relu(x_service)
                x_service = F.dropout(x_service, self.dropout, training=self.training)

        x_service = self.serviceLin(x_service)
        x = self.nodeLin(x)
        x = scatter(x, data.batch, dim=0, reduce='mean')
        serviceBatchList = []
        for i in range(x.size(0)):
            for j in range(self.outChannels):
                serviceBatchList.append(j)
        serviceBatch = torch.tensor(serviceBatchList, dtype=torch.long).to(torch.device('cuda'))
        x_service = scatter(x_service, serviceBatch, dim=0, reduce='mean')
        x_service = x_service.transpose(0, 1)
        x = torch.matmul(x, x_service)

        return self.sigmoid(x)
