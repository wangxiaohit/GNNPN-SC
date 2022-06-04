import json
from src.loadData import loadDataPN
import numpy as np


def calc(qos, cons):
    obj = 0.5 * (np.average(qos[0]) + 1 - np.min(qos[1]))
    if np.cumprod(qos[2])[-1] < cons[0][0] or np.cumprod(qos[2])[-1] > cons[0][1]:
        obj += 1
    if np.cumprod(qos[3])[-1] < cons[1][0] or np.cumprod(qos[3])[-1] > cons[1][1]:
        obj += 1
    return obj


def check(dataset, serCategory, epoch):
    tag = 0
    qosNum = 4

    newServiceFeatures, newlabels = loadDataPN(epoch=-1, dataset=dataset, serviceNumber=1)
    trainDataLen = len(newServiceFeatures) // 4 * 3
    testDataLen = len(newServiceFeatures) // 4

    with open(f"./data/{dataset}/minCostList.data", "r") as f:
        minCostList = json.load(f)
    if epoch == -1:
        url = f"./solutions/pretrained/{dataset}-PNHigh.txt"
    else:
        url = f"./solutions/PNHigh/{dataset}/allActions{epoch}.txt"

    with open(url) as f:
        allActions = json.load(f)

    allActionsSolution = [[0] * serCategory for _ in range(testDataLen)]
    for i in range(serCategory):
        for j in range(len(allActions[i])):
            allActionsSolution[j][i] = allActions[i][j][tag: tag + qosNum]

    newSolution = []
    for i in range(len(allActionsSolution)):
        _newSolution = []
        for action in allActionsSolution[i]:
            if sum(action) != 3:
                _newSolution.append(action)
        newSolution.append(_newSolution)

    all = 0
    times = 0
    for serviceFeatures, minCost, services in zip(newServiceFeatures[trainDataLen:], minCostList[trainDataLen:],
                                                  newSolution):
        times += 1
        cons = [serviceFeatures[0][qosNum + 1:][:2], serviceFeatures[0][qosNum + 1:][2:]]
        qos = []
        for i in range(qosNum):
            qos.append([services[j][i] for j in range(len(services))])
        all += minCost / calc(qos, cons)

    print(epoch, all / testDataLen)