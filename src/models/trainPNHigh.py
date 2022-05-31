import torch
import torch.optim as optim
import json
import time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from IPython.display import clear_output
import matplotlib.pyplot as plt

from modelPN import CombinatorialRL
from modelPN import reward
from loadData import loadDataPN


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
    def __init__(self, model, train_dataset, val_dataset, batch_size=128, threshold=None, max_grad_norm=2., low_model=None):
        self.model = model
        self.low_model = low_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.actor_optim = optim.Adam(model.actor.parameters(), lr=0.5e-4)  # all 0.5
        self.max_grad_norm = max_grad_norm

        self.train_tour = []
        self.val_tour = []

        self.epochs = 0

    def train_and_validate(self, n_epochs, epochDiv):
        critic_exp_mvg_avg = torch.zeros(1)
        if USE_CUDA:
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()
        latent = None
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
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

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
                torch.save(state, f"./solutionPN/{dataset}highRL/epoch{self.epochs // epochDiv}.model")
                state = {
                    "epoch": epoch,
                    "model": self.low_model.state_dict(),
                    "optimizer": self.actor_optim.state_dict()
                }
                torch.save(state, f"./solutionPN/{dataset}highRL/epoch{self.epochs // epochDiv}_low.model")

                self.model.eval()
                self.low_model.eval()
                allActions = [[] for _ in range(serCategory)]
                for val_batch, labs in self.val_loader:  # numbers
                    inputs = Variable(val_batch)
                    inputs = inputs.cuda()

                    _, _, _, _, latent = self.low_model(inputs, labs, sample="greedy", training="SL")  # numbers
                    R, probs, actions, actions_idxs, _ = self.model(inputs, labs, latent, sample="greedy")  # numbers
                    for a in range(len(actions)):
                        allActions[a] += actions[a].cpu().numpy().tolist()
                    self.val_tour.append(R.mean().item())
                with open(f"./solutionPN/{dataset}highRL/allActions{self.epochs // epochDiv}.txt", "w") as f:
                    json.dump(allActions, f)
                print((time.time() - t) / 1000)
                self.plot(self.epochs, epochDiv)
                with open(f"./solutionPN/{dataset}highRL/val{self.epochs // epochDiv}.txt", "w") as f:
                    json.dump(self.val_tour, f)
                with open(f"./solutionPN/{dataset}highRL/time{self.epochs // epochDiv}.txt", "w") as f:
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
        print(f"./solutionPN/{dataset}highRL/epoch{self.epochs // epochDiv}.png")
        plt.savefig(f"./solutionPN/{dataset}highRL/epoch{self.epochs // epochDiv}.png")
        plt.show()


if __name__ == "__main__":
    trainTag = "train"
    USE_CUDA = True
    embeddingTag = False
    para = True
    serCategory = 47
    epochsList = [9, 9, 9, 6, 9, 6, 6, 1]  # 4-10 qws 3 normal 1
    epoch = 4  # normal 2 qws 4
    epochDiv = 1
    for serNumber, epochs in zip(range(2, 5), epochsList[:3]):
        dataset = "qws1/"

        serviceFeatures, labels = loadDataPN(epoch=epoch, dataset=dataset[:-1], serviceNumber=serNumber)
        if para:
            dataset += f"paras/{serNumber}/"
        if embeddingTag:
            dataset += "20embeddings/"
            embedding_size = 20
        else:
            embedding_size = 0
        trainDataLen = len(serviceFeatures) // 4 * 3
        train_dataset = SCDataset(serviceFeatures[:trainDataLen], labels[:trainDataLen], embeddingTag)
        val_dataset = SCDataset(serviceFeatures[trainDataLen:], labels[trainDataLen:], embeddingTag)

        t = time.time()
        hidden_size = 256
        n_glimpses = 0
        tanh_exploration = 10
        use_tanh = True
        serviceNumber = serCategory * serNumber + 0  # 0->4

        beta = 0.9
        max_grad_norm = 2.

        model_low = CombinatorialRL(
            embedding_size,
            hidden_size,
            serviceNumber,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            reward,
            attention="Dot",
            use_cuda=USE_CUDA,
            sNumber=serNumber,
            sCategory=serCategory)

        model_high = CombinatorialRL(
            embedding_size,
            hidden_size,
            serviceNumber,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            reward,
            attention="Dot",
            use_cuda=USE_CUDA,
            level="High",
            sNumber=serNumber,
            sCategory=serCategory
        )

        if trainTag == "train":
            load_root = f"./solutionPN/{dataset}low/epoch{epochs}.model"
            # load_root = f"./solutionPN/{dataset}low/lowPN.model"
            state = torch.load(load_root)
            model_low.load_state_dict(state['model'])

        else:
            # dataset = f"dataset1/paras/{serNumber}/"

            load_root = f"./solutionPN/{dataset}highRL/epoch299_low.model"
            state = torch.load(load_root)
            model_low.load_state_dict(state['model'])

            load_root = f"./solutionPN/{dataset}highRL/epoch299.model"
            state = torch.load(load_root)
            model_high.load_state_dict(state['model'])
            # dataset = f"dataset1/paras/{serNumber}/"

        if USE_CUDA:
            model_low = model_low.cuda()
            model_high = model_high.cuda()
        model_high = TrainModel(model_high,
                                train_dataset,
                                val_dataset,
                                low_model=model_low)

        model_high.train_and_validate(100, epochDiv)
