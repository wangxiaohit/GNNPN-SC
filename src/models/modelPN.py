import math
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


qosandcons = 8
qosNum = 4
consNum = 2


def calc(services, constraints, sCategory):
    violate = 0
    serviceNum = 0
    violateConstraints = []
    indicator = [np.array([services[i][j].cpu() for i in range(len(services))]) for j in range(qosNum)]
    conValues = [np.cumprod(indicator[i + 2])[-1] for i in range(consNum)]
    for i in range(len(constraints)):
        for constraint in constraints[i]:
            if conValues[i] < constraint[-2] or conValues[i] > constraint[-1]:
                violate += 1
                violateConstraints.append([i, constraint])
    for i in range(sCategory):
        if services[i][0] > 0:
            serviceNum += 1
    objFunc = (np.sum(indicator[0]) / serviceNum + 1 - np.min(indicator[1])) / 2
    objFunc = float(objFunc)

    return violate, objFunc, violateConstraints


def reward(sample_solution, optSolutions, sCategory, USE_CUDA=False, level="Low", embedding_size=20):
    """
    Args:
        sample_solution seq_len of [batch_size]
    """
    batch_size = sample_solution[0].size(0)
    optList = [0] * batch_size
    if embedding_size == 0:
        tag = 0
    else:
        tag = 1
    for j in range(len(sample_solution[0])):
        constraintsList = [[] for _ in range(consNum)]
        serviceList = [[]] * sCategory
        for i in range(len(sample_solution)):
            serviceNumber = i
            if serviceNumber == 0:
                for kk in range(consNum):
                    constraintsList[kk].append(
                        [sample_solution[i][j][tag + qosNum + kk * 2].item(), sample_solution[i][j][tag + 1 + qosNum + kk * 2].item()])
            serviceList[serviceNumber] = sample_solution[i][j][tag: tag + qosNum]

        violate, objFunc, violateConstraints = calc(serviceList, constraintsList, sCategory)
        if level == "Low":
            optList[j] = violate
        else:
            optList[j] = round(violate + objFunc, 5)
    sumVio = 0
    avg = np.average(optList)
    for i in optList:
        if i >= 1:
            sumVio += 1
    print(f"{level}, {sumVio}, {avg}: ", optList)
    optList = torch.FloatTensor(optList)
    if USE_CUDA:
        optList = optList.cuda()

    return optList


class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau', use_cuda=True):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            if use_cuda:
                V = V.cuda()
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, torch.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)

        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * torch.tanh(logits)
        else:
            logits = logits
        return ref, logits


class PointerNet(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 attention,
                 sNumber,
                 sCategory,
                 use_cuda=True,
                 level="low",
                 mask=False):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len
        self.use_cuda = use_cuda
        self.level = level
        self.serNumber = sNumber
        self.serCategory = sCategory

        self.alpha = torch.ones(1).cuda()
        self.mask = mask
        if embedding_size != 0:
            self.embedding1 = nn.Embedding(self.serCategory, embedding_size)
        self.embedding2 = nn.Linear(embedding_size + qosandcons, hidden_size)

        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention, use_cuda=use_cuda)
        self.glimpse = Attention(hidden_size, use_tanh=False, name=attention, use_cuda=use_cuda)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(hidden_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            clone_mask = clone_mask.bool()
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self, inputs, latent, sample="sample"):
        """
        Args:
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        assert seq_len == self.seq_len
        if self.embedding_size != 0:
            x1 = inputs[:, :, 0].long()
            x2 = inputs[:, :, 1:]
            x1 = self.embedding1(x1)
            embedded = torch.cat((x1, x2), 2)
        else:
            embedded = inputs.clone()
        embedded = self.embedding2(embedded)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_probs = []
        prev_idxs = []
        prev_logits = []
        mask = torch.zeros(batch_size, seq_len).byte()
        if self.use_cuda:
            mask = mask.cuda()

        idxs = None

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for k in range(self.serCategory):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            if latent:
                logits_new = logits + self.alpha * latent[k]
            else:
                logits_new = logits.clone()

            for p in range(len(logits_new)):
                logits_new[p][: k * self.serNumber] = -np.inf
                logits_new[p][(k + 1) * self.serNumber:] = -np.inf

            probs = F.softmax(logits_new, dim=1)
            if sample == "greedy":
                _, idxs = torch.max(probs, dim=1)
            else:
                idxs = probs.multinomial(num_samples=1).squeeze(1)
            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    print(seq_len)
                    print(' RESAMPLE!')
                    idxs = probs.multinomial(num_samples=1).squeeze(1)
                    break
            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :]

            prev_probs.append(probs)
            prev_idxs.append(idxs)
            prev_logits.append(logits)

        return prev_probs, prev_idxs, prev_logits


class CombinatorialRL(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 reward,
                 attention,
                 sNumber,
                 sCategory,
                 use_cuda=True,
                 level="Low",
                 mask=False,):
        super(CombinatorialRL, self).__init__()
        self.reward = reward
        self.use_cuda = use_cuda
        self.level = level
        self.embedding_size = embedding_size
        self.sNumber = sNumber
        self.serCategory = sCategory

        self.actor = PointerNet(
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            attention,
            sNumber,
            sCategory,
            use_cuda,
            level=level,
            mask=mask,
        )

    def forward(self, inputs, labs, latent=None, sample="sample", training="RL"):  # numbers
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch_size = inputs.size(0)
        input_size = inputs.size(1)
        seq_len = inputs.size(2)
        probs, action_idxs, logits = self.actor(inputs, latent, sample=sample)
        latent_p = logits.copy()

        actions = []
        for action_id in action_idxs:
            actions.append(inputs[[x for x in range(batch_size)], action_id.data, :])

        action_probs = []
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[[x for x in range(batch_size)], action_id.data])

        if training == "RL":
            R = self.reward(actions, labs, self.serCategory,
                            USE_CUDA=self.use_cuda, level=self.level, embedding_size=self.embedding_size)
            return R, action_probs, actions, action_idxs, latent_p
        else:
            return probs, action_probs, actions, action_idxs, latent_p
