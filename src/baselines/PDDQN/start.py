from src.baselines.PDDQN.dueling_ddqn import DuelingAgent
from src.loadData import loadDataPN
import time
import numpy as np
import json
from src.baselines.PDDQN.model import DuelingDQN
import torch


class SC:
    def __init__(self, actions, constraints, serviceCategory, serviceNumber):
        self.action_space = actions
        self.observation_space = 8
        self.serviceCategory = serviceCategory
        self.serviceNumber = serviceNumber
        self.constraints = constraints
        self.qosNum = 4
        self.consNum = 2

    def reset(self):
        return [0, 1, 1, 1, 0, 0, 0, 0]

    def sample(self):
        idx = np.random.choice(self.serviceNumber)
        return idx

    def step(self, state, action, number):
        service = self.action_space[number][action]
        state[0] = (state[0] * number + service[0]) / (number + 1)
        state[1] = min(state[1], service[1])
        state[2] *= service[2]
        state[3] *= service[3]
        state[self.qosNum:] = service[:4]
        number += 1
        reward = 1 - (service[0] + 1 - service[1])

        if number == self.serviceCategory:
            v = 0
            if not self.constraints[0][0] <= state[2] <= self.constraints[0][1]:
                v += 1
            if not self.constraints[1][0] <= state[3] <= self.constraints[1][1]:
                v += 1
            o = (state[0] + 1 - state[1]) / 2
            reward = 1 - (v + o)
        return state, reward, number


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size, serviceCategory):
    episode_rewards = []
    best = 3
    eps = [0.2] * max_episodes + [1]
    bufferNum = 0
    for episode in range(max_episodes + 1):
        state = agent.env.reset()
        number = 0
        episode_reward = 0
        for step in range(max_steps):
            action = agent.get_action(state, eps=eps[episode])
            next_state, reward, number = env.step(state, action, number)
            agent.replay_buffer.push(state, action, reward, next_state)
            bufferNum += 1
            episode_reward += reward

            if bufferNum > batch_size:
                agent.update(batch_size)
                bufferNum = 0

            if number == serviceCategory:
                episode_rewards.append(episode_reward)
                if 1 - reward < best:
                    best = 1 - reward
                break

            state = next_state

    return best


class PDDQN:
    def __init__(self, dataset, maxEpisodes, batchSize, serviceCategory, serviceNumber, epoch):
        self.dataset = dataset + "/"
        self.MAX_EPISODES = maxEpisodes
        self.BATCH_SIZE = batchSize
        self.serviceCategory = serviceCategory
        self.serviceNumber = serviceNumber
        self.epoch = epoch
        self.n_epochs = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def start(self):
        newServiceFeatures, _ = loadDataPN(epoch=self.epoch, dataset=self.dataset[:-1],
                                           serviceNumber=self.serviceNumber)
        with open(f"./data/{self.dataset}minCostList.data", "r") as f:
            minCostList = json.load(f)

        times = 0
        url = f"./solutions/WOA/{self.dataset}/ML+PDDQN.txt"
        qualities = {
            "quality": [],
            "time": [],
            "averageQ": 0,
            "averageT": 0
        }
        actionsList = []
        constraintsList = []
        for newServiceFeature, minCost in zip(newServiceFeatures, minCostList):
            actions = []
            idx = 0
            for i in range(self.serviceCategory):
                _actions = []
                for j in range(self.serviceNumber):
                    _actions.append(newServiceFeature[idx][1: 5])
                    idx += 1
                if _actions[0] != [0, 1, 1, 1]:
                    actions.append(_actions)
            actionsList.append(actions)
            constraintsList.append([newServiceFeature[0][5: 7], newServiceFeature[0][7:]])

        for actions, constraints, minCost in zip(actionsList[len(actionsList) // 4 * 3:],
                                                 constraintsList[len(actionsList) // 4 * 3:],
                                                 minCostList[len(actionsList) // 4 * 3:]):
            times += 1
            SCmodel = SC(actions, constraints, len(actions), self.serviceNumber)
            agent = DuelingAgent(SCmodel, use_conv=False)
            t = time.time()
            q = mini_batch_train(SCmodel, agent, self.MAX_EPISODES, len(actions), self.BATCH_SIZE, len(actions))
            tt = time.time() - t
            qualities["quality"].append(minCost / q)
            qualities["time"].append(tt)
            qualities["averageQ"] = np.average(qualities["quality"])
            qualities["averageT"] = np.average(qualities["time"])
            print(times, np.average(qualities["quality"]), np.average(qualities["time"]))

        with open(url, "w") as f:
            json.dump(qualities, f)
