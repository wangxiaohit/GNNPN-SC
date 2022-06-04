import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCELoss
import os.path as osp
import json
import time
import math
import numpy as np

from torch_geometric.data import Data, Dataset, DataLoader
from src.models.modelML import Net
from src.loadData import loadData


class TrainML:
    def __init__(self, dataset1, numLayersGIN, numLayersGCN, hiddenChannels, embeddingChannels, dropout, lr, epochs):
        self.dataset1 = dataset1
        self.hiddenChannels = hiddenChannels
        self.embeddingChannels = embeddingChannels
        self.numLayersGIN = numLayersGIN
        self.numLayersGCN = numLayersGCN
        self.epochs = epochs
        self.dropout = dropout
        self.lr = lr

        self.device = torch.device('cuda')
        self.criterion = BCELoss()
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None

    def train(self):
        self.model.train()

        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = self.model(data).squeeze()
            loss = self.criterion(x, data.y.view(x.size(0), x.size(1)))
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            self.optimizer.step()

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        idxList = []
        log = [1, 5]

        total_pat = [[] for _ in range(len(log))]
        for data in loader:
            data = data.to(self.device)
            x = self.model(data).squeeze()
            y = data.y.view(x.size(0), x.size(1))
            for _x, _y in zip(x, y):
                pat = [0] * len(log)
                srt1, indices = _x.sort(dim=0, descending=True)
                for k in range(len(log)):
                    for i, idx in zip(range(1, log[k] + 1), indices[:log[k]]):
                        if _y[idx] == 1:
                            pat[k] += 1

                idxList.append(indices.cpu().numpy().tolist())
                for k in range(len(log)):
                    total_pat[k].append(pat[k] / log[k])

        return idxList, [np.average(total_pat[k]) for k in range(len(log))]

    def start(self):

        class CompositionDataset(Dataset):
            def __init__(self, root, transform=None, pre_transform=None):
                super(CompositionDataset, self).__init__(root, transform, pre_transform)

            @property
            def raw_file_names(self):
                return []

            @property
            def processed_file_names(self):
                return ['data_{}.pt'.format(i) for i in range(len(nodefeatures))]

            def download(self):
                pass

            def process(self):
                for i in range(len(nodefeatures)):
                    data = Data(x=torch.tensor(nodefeatures[i], dtype=torch.float),
                                y=torch.tensor(labels[i], dtype=torch.float),
                                edge_index=torch.tensor(edge_indices[i], dtype=torch.long))

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(data, osp.join(self.processed_dir, "data_{}.pt".format(i)))

            def len(self):
                return len(self.processed_file_names)

            def get(self, idx):
                data = torch.load(osp.join(self.processed_dir, "data_{}.pt".format(idx)))
                return data

        class AddServiceMap(object):
            def __call__(self, data):
                data.x_service = torch.tensor(serviceFeatureList, dtype=torch.float)
                data.edge_index_service = torch.tensor(edge_indices_service, dtype=torch.long)
                data.edge_attr_service = torch.tensor(edge_attrs_service, dtype=torch.float)
                return data

        nodefeatures, serviceFeatureList, edge_indices, edge_indices_service, edge_attrs_service, labels, inv_psp = loadData(self.dataset1)
        self.dataset1 += "/"
        transform = AddServiceMap()

        dataset = CompositionDataset(root=f"./dataset/{self.dataset1}", pre_transform=transform)
        self.train_loader = DataLoader(dataset[: len(nodefeatures) // 4 * 3], batch_size=2, shuffle=True)
        self.val_loader = DataLoader(dataset[len(nodefeatures) // 4 * 3:], batch_size=2)
        t = time.time()

        self.model = Net(hiddenChannels=self.hiddenChannels, outChannels=len(labels[0]), embeddingChannels=self.embeddingChannels, numLayersGIN=self.numLayersGIN,
                         numLayersGCN=self.numLayersGCN, isServices=True, dropout=0.0).to(self.device)

        print()
        print(f"Run {0}:")
        print()

        self.model.reset_parameters()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                      patience=3, min_lr=0.00001)

        for epoch in range(self.epochs):
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss = self.train()
            val_idxList, val_mae = self.test(self.val_loader)
            scheduler.step(val_mae[0])

            print(f"Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.4f}, ValP@1: {val_mae[0]:.4f}, ValP@5: {val_mae[1]:.4f}")
            print(time.time() - t)

            test_idxList, test_mae = self.test(self.train_loader)
            torch.save(self.model, f"solutions/ML/{self.dataset1}model-{epoch}.pkl")
            with open(f"solutions/ML/{self.dataset1}testServices-epoch{epoch}.txt", "w") as f:
                json.dump(test_idxList + val_idxList, f)

