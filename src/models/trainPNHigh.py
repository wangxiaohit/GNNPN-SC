import torch
import torch.optim as optim
import json
import time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from IPython.display import clear_output
import matplotlib.pyplot as plt

from src.models.modelPN import reward, CombinatorialRL
from src.loadData import loadDataPN


class SCDataset(Dataset):

    def __init__(self, dataset, targets, embeddingTag=False):
        super(SCDataset, self).__init__()

        self.data_set = []
        self.label = []
        self.serviceNumbers = []
        for data, target in zip(dataset, targets):
            _data = []
            for i in range(len(data)):
                if not embeddingTag:
                    _data.append(data[i][1:])
                else:
                    _data.append(data[i])

            self.data_set.append(torch.FloatTensor(_data))
            self.label.append(target)
            self.serviceNumbers.append(0)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx], self.label[idx]


class TrainModel:
    def __init__(self, model, train_dataset, val_dataset, epochDiv, beta, USE_CUDA, dataset, serCategory, lr=0.5e-4,
                 batch_size=128, threshold=None, max_grad_norm=2., low_model=None):
        self.model = model
        self.low_model = low_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        self.epochDiv = epochDiv
        self.beta = beta
        self.USE_CUDA = USE_CUDA
        self.dataset = dataset
        self.serCategory = serCategory

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.actor_optim = optim.Adam(model.actor.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm

        self.train_tour = []
        self.val_tour = []

        self.epochs = 0

    def train_and_validate(self, n_epochs, epochDiv):
        critic_exp_mvg_avg = torch.zeros(1)
        if self.USE_CUDA:
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()
        latent = None
        t = time.time()
        for epoch in range(1, n_epochs + 1):
            for batch_id, (sample_batch, labs) in enumerate(self.train_loader):
                self.low_model.train()
                self.model.train()
                inputs = Variable(sample_batch)
                inputs = inputs.cuda()

                _, _, _, _, latent = self.low_model(inputs, labs, sample="greedy", training="SL")
                R, probs, actions, actions_idxs, _ = self.model(inputs, labs, latent)

                # RL
                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * self.beta) + ((1. - self.beta) * R.mean())

                advantage = R - critic_exp_mvg_avg

                logprobs = 0
                for prob in probs:
                    logprob = torch.log(prob)
                    logprobs += logprob
                logprobs[logprobs < -1000] = 0.

                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(),
                                               float(self.max_grad_norm), norm_type=2)

                self.actor_optim.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                self.train_tour.append(R.mean().item())

            if self.threshold and self.train_tour[-1] < self.threshold:
                print("EARLY STOPPAGE!")
                break
            if epoch % epochDiv == 0:
                state = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.actor_optim.state_dict()
                }
                torch.save(state, f"./solutions/PNHigh/{self.dataset}/epoch{self.epochs // epochDiv}.model")
                state = {
                    "epoch": epoch,
                    "model": self.low_model.state_dict(),
                    "optimizer": self.actor_optim.state_dict()
                }
                torch.save(state, f"./solutions/PNHigh/{self.dataset}/epoch{self.epochs // epochDiv}_low.model")

                self.model.eval()
                self.low_model.eval()
                allActions = [[] for _ in range(self.serCategory)]
                for val_batch, labs in self.val_loader:
                    inputs = Variable(val_batch)
                    inputs = inputs.cuda()

                    _, _, _, _, latent = self.low_model(inputs, labs, sample="greedy", training="SL")  # numbers
                    R, probs, actions, actions_idxs, _ = self.model(inputs, labs, latent, sample="greedy")  # numbers
                    for a in range(len(actions)):
                        allActions[a] += actions[a].cpu().numpy().tolist()
                    self.val_tour.append(R.mean().item())
                with open(f"./solutions/PNHigh/{self.dataset}allActions{self.epochs // epochDiv}.txt", "w") as f:
                    json.dump(allActions, f)
                print((time.time() - t))
                self.plot(self.epochs, epochDiv)
                with open(f"./solutions/PNHigh/{self.dataset}/val{self.epochs // epochDiv}.txt", "w") as f:
                    json.dump(self.val_tour, f)
                with open(f"./solutions/PNHigh/{self.dataset}/time{self.epochs // epochDiv}.txt", "w") as f:
                    json.dump([time.time() - t], f)
            self.epochs += 1

    def plot(self, epoch, epochDiv):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('optTarget: epoch %s reward %s' %
                  (epoch // epochDiv, self.train_tour[-1] if len(self.train_tour) else 'collecting'))
        if len(self.train_tour) > 2000:
            plt.plot(self.train_tour[-2000:])
        else:
            plt.plot(self.train_tour)
        # plt.plot(self.train_tour)
        plt.grid()
        plt.subplot(132)
        plt.title('optTarget: epoch %s reward %s' %
                  (epoch // epochDiv, self.val_tour[-1] if len(self.val_tour) else 'collecting'))
        plt.plot(self.val_tour)
        plt.grid()
        print(f"./solutions/PNHigh/{self.dataset}/epoch{self.epochs // epochDiv}.png")
        plt.savefig(f"./solutions/PNHigh/{self.dataset}/epoch{self.epochs // epochDiv}.png")
        # plt.show()


class PNHigh:
    def __init__(self, dataset, embeddingTag, USE_CUDA, serCategory, epochDiv, serNumber, hidden_size, n_glimpses,
                 tanh_exploration, use_tanh, beta, max_grad_norm, lr, epochML, epochPNLow):
        self.dataset = dataset + "/"
        self.embeddingTag = embeddingTag
        self.USE_CUDA = USE_CUDA
        self.serCategory = serCategory
        self.epochDiv = epochDiv
        self.serNumber = serNumber
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.tanh_exploration = tanh_exploration
        self.use_tanh = use_tanh
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.epochML = epochML
        self.epochPNLow = epochPNLow

    def start(self):

        serviceFeatures, labels = loadDataPN(epoch=self.epochML, dataset=self.dataset[:-1], serviceNumber=self.serNumber)
        if self.embeddingTag:
            self.dataset += "20embeddings/"
            embedding_size = 20
        else:
            embedding_size = 0
        trainDataLen = len(serviceFeatures) // 4 * 3
        train_dataset = SCDataset(serviceFeatures[:trainDataLen], labels[:trainDataLen], self.embeddingTag)
        val_dataset = SCDataset(serviceFeatures[trainDataLen:], labels[trainDataLen:], self.embeddingTag)

        serviceNumber = self.serCategory * self.serNumber

        model_low = CombinatorialRL(
            embedding_size,
            self.hidden_size,
            serviceNumber,
            self.n_glimpses,
            self.tanh_exploration,
            self.use_tanh,
            reward,
            attention="Dot",
            level="Low",
            use_cuda=self.USE_CUDA,
            sNumber=self.serNumber,
            sCategory=self.serCategory
        )

        model_high = CombinatorialRL(
            embedding_size,
            self.hidden_size,
            serviceNumber,
            self.n_glimpses,
            self.tanh_exploration,
            self.use_tanh,
            reward,
            attention="Dot",
            level="High",
            use_cuda=self.USE_CUDA,
            sNumber=self.serNumber,
            sCategory=self.serCategory
        )
        if self.epochPNLow >= 0:
            load_root = f"./solutions/PNLow/{self.dataset}/epoch{self.epochPNLow}.model"
        else:
            load_root = f"./solutions/pretrained/{self.dataset[:-1]}-PNLow.model"
        state = torch.load(load_root)
        model_low.load_state_dict(state['model'])

        if self.USE_CUDA:
            model_low = model_low.cuda()
            model_high = model_high.cuda()
        model_high = TrainModel(model_high, train_dataset, val_dataset, self.epochDiv, self.beta, self.USE_CUDA,
                                self.dataset, self.serCategory, self.lr, 128, None, self.max_grad_norm,
                                low_model=model_low)

        model_high.train_and_validate(100, self.epochDiv)
