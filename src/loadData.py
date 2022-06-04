import json
import numpy as np
from tqdm import tqdm


def compute_inv_propesity(labels, A, B):
    num_instances = len(labels)
    freqs = np.ravel(np.sum(labels, axis=0))
    C = (np.log(num_instances)-1)*np.power(B+1, A)
    wts = 1.0 + C*np.power(freqs+B, -A)
    return np.ravel(wts)


def loadData(dataset=""):
    if dataset != "":
        dataset = dataset + "/"
    with open(f"./data/{dataset}nodefeatures.data", "r") as f:
        nodefeatures = json.load(f)
    with open(f"./data/{dataset}edge_indices.data", "r") as f:
        edge_indices = json.load(f)
    with open(f"./data/{dataset}labels.data", "r") as f:
        labels = json.load(f)
    with open(f"./data/{dataset}serviceFeature.data", "r") as f:
        serviceFeature = json.load(f)

    nodeFeaturesNew = []
    for nodefeature in nodefeatures:
        nodeFeatureNew = []
        for feature in nodefeature:
            featureType = feature[: -6].index(1)
            nodeFeatureNew.append([featureType] + feature[-6:])
        nodeFeaturesNew.append(nodeFeatureNew)
    nodefeatures = nodeFeaturesNew

    serviceFeatureList = []
    keys = sorted(map(eval, serviceFeature.keys()))
    for key in keys:
        features = serviceFeature[str(key)]
        for feature in features:
            serviceFeatureList.append([key - int(keys[0])] + feature[-4:])

    adjMat = [[0 for _ in range(len(labels[0]))] for _ in range(len(labels[0]))]
    serviceUseTimes = [0 for _ in range(len(labels[0]))]
    for label in labels[:3000]:
        couses = []
        for lab in range(len(label)):
            if label[lab] == 1:
                serviceUseTimes[lab] += 1
                couses.append(lab)
        for i in range(len(couses) - 1):
            for j in range(i + 1, len(couses)):
                adjMat[couses[i]][couses[j]] += 1
                adjMat[couses[j]][couses[i]] += 1

    edge_indices_service = [[], []]
    edge_attrs_service = []
    for i in range(len(adjMat) - 1):
        for j in range(i + 1, len(adjMat)):
            if adjMat[i][j] != 0:
                edge_indices_service[0].append(i)
                edge_indices_service[0].append(j)
                edge_indices_service[1].append(j)
                edge_indices_service[1].append(i)
                edge_attrs_service.append(adjMat[i][j] / serviceUseTimes[i])
                edge_attrs_service.append(adjMat[j][i] / serviceUseTimes[j])

    inv_propen = compute_inv_propesity(labels[:3000], 0.55, 1.5)

    return nodefeatures, serviceFeatureList, edge_indices, edge_indices_service, edge_attrs_service, labels, inv_propen


def loadDataPN(epoch=7, dataset="", serviceNumber=5):
    if dataset != "":
        dataset = dataset + "/"
    with open(f"./data/{dataset}nodefeatures.data", "r") as f:
        nodefeatures = json.load(f)
    with open(f"./data/{dataset}labels.data", "r") as f:
        labels = json.load(f)
    with open(f"./data/{dataset}serviceFeature.data", "r") as f:
        serviceFeature = json.load(f)
    with open(f"./data/{dataset}minCostList.data", "r") as f:
        minCostList = json.load(f)

    if epoch >= 0:
        with open(f"./solutions/ML/{dataset}testServices-epoch{epoch}.txt", "r") as f:
            testServices = json.load(f)
    else:
        with open(f"./solutions/pretrained/{dataset[:-1]}-ML.txt", "r") as f:
            testServices = json.load(f)

    serCategory = len(serviceFeature.keys())
    ser2idxdiv = []
    ser2idxmod = []
    for key in serviceFeature.keys():
        index = int(key) - 1
        ser2idxdiv += [index] * len(serviceFeature[key])
        ser2idxmod += [i for i in range(len(serviceFeature[key]))]

    newServiceFeatures = []
    newlabels = []
    for nodefeature, label, testService, minCost in zip(nodefeatures, labels, testServices, minCostList):
        constraints = dict()
        serviceSet = set()
        for i in range(1, serCategory + 1):
            constraints[i] = [0] * 8

        for node in nodefeature:
            if node[0] == 1:
                for i in range(1, serCategory + 1):
                    constraints[i][-4:] = node[-5: -3] + node[-2:]
            else:
                idx = node[:-6].index(1)
                constraints[idx][-8: -4] = node[-5: -3] + node[-2:]
                serviceSet.add(idx)

        serviceFiveSets = [set() for _ in range(serCategory)]
        for s in testService[: len(testService)]:
            if len(serviceFiveSets[ser2idxdiv[s]]) < serviceNumber:
                serIdx = str(ser2idxdiv[s] + 1)
                serCost = serviceFeature[serIdx][ser2idxmod[s]][-2]
                serQuality = serviceFeature[serIdx][ser2idxmod[s]][-1]
                serIdx = int(serIdx)
                if constraints[serIdx][-8] <= serCost <= constraints[serIdx][-7] \
                        and constraints[serIdx][-6] <= serQuality <= constraints[serIdx][-5]:
                    serviceFiveSets[ser2idxdiv[s]].add(s)
        newServiceFeature = []
        ser1, ser2 = [], []
        for i in range(len(serviceFiveSets)):
            key = i + 1
            if i == 0:
                x = constraints[int(key)][-4:]
            else:
                x = [0, 0, 0, 0]
            serviceFiveSets[i] = list(serviceFiveSets[i])
            np.random.shuffle(serviceFiveSets[i])
            if key in serviceSet:
                while len(serviceFiveSets[i]) < serviceNumber:
                    serviceFiveSets[i] += serviceFiveSets[i]
                newServiceFeature += [
                    [i] + [serviceFeature[str(key)][ser2idxmod[v]][k] for k in [-4, -3, -2, -1]] + x for v in
                    serviceFiveSets[i][:serviceNumber]]
                ser1.append([serviceFeature[str(key)][ser2idxmod[v]][-4] for v in serviceFiveSets[i][:serviceNumber]])
                ser2.append([serviceFeature[str(key)][ser2idxmod[v]][-3] for v in serviceFiveSets[i][:serviceNumber]])
                if len(serviceFiveSets[i]) == 0:
                    print(serviceFiveSets[i])

            else:
                newServiceFeature += [[i, 0, 1, 1, 1] + x for _ in range(serviceNumber)]
        newServiceFeatures.append(newServiceFeature)
        newlabels.append(minCost)

    return newServiceFeatures, newlabels


def addS(PriS, serviceFeatures, constraints, serviceIndex, ser2idxdiv, ser2idxmod, reduct=False, sSet=None):
    serCategory = 50
    PriSNew = [[] for _ in range(serCategory)]
    min0, min1, min2, min3 = [[1] for _ in range(serCategory)], [[0] for _ in range(serCategory)],\
                             [[1] for _ in range(serCategory)], [[1] for _ in range(serCategory)]
    for s in PriS:
        serIdx = str(ser2idxdiv[s] + 1)
        ser0 = serviceFeatures[serIdx][ser2idxmod[s]][-4]
        ser1 = serviceFeatures[serIdx][ser2idxmod[s]][-3]
        serCost = serviceFeatures[serIdx][ser2idxmod[s]][-2]
        serQuality = serviceFeatures[serIdx][ser2idxmod[s]][-1]
        serIdx = int(serIdx)

        if constraints[serIdx][0] <= serCost <= constraints[serIdx][1] and constraints[serIdx][2] <= serQuality <= \
                constraints[serIdx][3]:
            if reduct:
                temp = 0
                for x in range(len(min0[serIdx - 1])):
                    roundService = tuple([round(min0[serIdx - 1][x], 5), round(min1[serIdx - 1][x], 5),
                                          round(min2[serIdx - 1][x], 5), round(min3[serIdx - 1][x], 5)])
                    if sSet and roundService in sSet:
                        continue
                    if ser0 < min0[serIdx - 1][x] and ser1 > min1[serIdx - 1][x] and min1[serIdx - 1][x] < reduct:
                        min0[serIdx - 1][x] = ser0
                        min1[serIdx - 1][x] = ser1
                        min2[serIdx - 1][x] = serCost
                        min3[serIdx - 1][x] = serQuality

                        if len(PriSNew[ser2idxdiv[s]]) == 0:
                            PriSNew[ser2idxdiv[s]].append(tuple([ser0, ser1, serCost, serQuality]))
                        else:
                            PriSNew[ser2idxdiv[s]][x] = tuple([ser0, ser1, serCost, serQuality])
                        temp = 1
                        break

                    if (ser0 > min0[serIdx - 1][x] and ser1 < min1[serIdx - 1][x]) or ser1 > reduct > ser0:
                        break
                roundService = tuple([round(ser0, 5), round(ser1, 5), round(serCost, 5), round(serQuality, 5)])
                if not temp and ((sSet and roundService in sSet) or ser1 > reduct > ser0):
                    min0[serIdx - 1].append(ser0)
                    min1[serIdx - 1].append(ser1)
                    min2[serIdx - 1].append(serCost)
                    min3[serIdx - 1].append(serQuality)
                    PriSNew[ser2idxdiv[s]].append(tuple([ser0, ser1, serCost, serQuality]))
            else:
                PriSNew[ser2idxdiv[s]].append(tuple([ser0, ser1, serCost, serQuality]))
    _PriSNew = [PriSNew[s] for s in serviceIndex]
    return _PriSNew


def loadDataOther(dataset="", reduct=False, sSetList=None, train=False):

    if dataset != "":
        dataset = dataset + "/"
    with open(f"./data/{dataset}nodefeatures.data", "r") as f:
        nodefeatures = json.load(f)
    with open(f"./data/{dataset}labels.data", "r") as f:
        labels = json.load(f)
    with open(f"./data/{dataset}serviceFeature.data", "r") as f:
        serviceFeature = json.load(f)
    with open(f"./data/{dataset}minCostList.data", "r") as f:
        minCostList = json.load(f)

    serCategory = len(serviceFeature.keys())
    ser2idxdiv = []
    ser2idxmod = []
    for key in serviceFeature.keys():
        index = int(key) - 1
        ser2idxdiv += [index] * len(serviceFeature[key])
        ser2idxmod += [i for i in range(len(serviceFeature[key]))]

    newServiceFeatures = []
    newlabels = []
    constraintsList = []
    if not train:
        left = len(nodefeatures) // 4 * 3
        _idx = len(nodefeatures) // 4 * 3
    else:
        left = 0
        _idx = 0

    print("Loading data...")
    for nodefeature, label in tqdm(zip(nodefeatures[left:], labels[left:])):
        constraints = dict()
        serviceSet = set()
        for i in range(1, serCategory + 1):
            constraints[i] = [0] * 8

        for node in nodefeature:
            if node[0] == 1:
                for i in range(1, serCategory + 1):
                    constraints[i][-4:] = node[-5: -3] + node[-2:]
            else:
                idx = node[:-6].index(1)
                constraints[idx][-8: -4] = node[-5: -3] + node[-2:]
                serviceSet.add(idx)
        nodeFeatureNew = []
        for feature in nodefeature:
            featureType = feature[: -6].index(1)
            nodeFeatureNew.append([featureType] + feature[-6:])
        nodefeature = nodeFeatureNew
        serviceIndex = [i[0] - 1 for i in nodefeature]
        serviceIndex = serviceIndex[1:]
        if sSetList and _idx >= len(nodefeatures) // 4 * 3:
            newServiceFeature = addS(list(range(len(ser2idxdiv))), serviceFeature, constraints, serviceIndex, ser2idxdiv, ser2idxmod, reduct, sSetList[_idx - len(nodefeatures) // 4 * 3])
        else:
            newServiceFeature = addS(list(range(len(ser2idxdiv))), serviceFeature, constraints, serviceIndex, ser2idxdiv, ser2idxmod, reduct)
        _newServiceFeature = []
        for feature in newServiceFeature:
            if len(feature) > 0:
                _newServiceFeature.append(feature)
        newServiceFeatures.append(_newServiceFeature)

        constraint = [[] for _ in range(2)]
        for key, value in constraints.items():
            constraint[0].append(value[-4: -2])
            constraint[1].append(value[-2:])
            break
        constraintsList.append(constraint)
        _idx += 1

    return newServiceFeatures, constraintsList, minCostList
