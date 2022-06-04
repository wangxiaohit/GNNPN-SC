import json
import time
import numpy as np
import math
import src.baselines.DPKSD.mine as mine


class GA:
    def __init__(self, constraints, services, popSize, stop, crossoverRate=0.8, mutationRate=0.2, qosNum=4):
        self.stop = 0
        self.constraints = constraints
        self.services = services
        self.mutaPointList = []
        for s in range(len(self.services)):
            if len(self.services[s]) > 1:
                self.mutaPointList.append(s)
        self.bestServices = []
        self.bestViolate = 0x7777777
        self.bestViolateConstraints = []
        self.bestObjFunc = 0x7777777
        self.nowServices = []

        self.qosNum = qosNum
        self.consNum = 2

        self.popSize = popSize
        self.stopEnd = stop
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.nGenerations = 999999

    def calc(self, services):
        violate = 0
        serviceNum = 0
        violateConstraints = []
        indicator = [np.array([services[i][j] for i in range(len(services))]) for j in
                     range(self.qosNum)]
        conValues = [np.cumprod(indicator[i + 2])[-1] for i in range(self.consNum)]
        for i in range(len(self.constraints)):
            for constraint in self.constraints[i]:
                if conValues[i] < constraint[-2] or conValues[i] > constraint[-1]:
                    violate += 1
                    violateConstraints.append([i, constraint])
        for i in range(len(services)):
            if services[i][0] > 0:
                serviceNum += 1

        objFunc = (np.sum(indicator[0]) / serviceNum + 1 - np.min(indicator[1])) / 2
        objFunc = float(objFunc)
        return violate, objFunc, violateConstraints

    def getFitness(self, pop, n):
        fitness = []
        for p in pop:
            violate, objFunc, violateConstraints = self.calc(p)
            if (violate < self.bestViolate) or (violate == self.bestViolate and objFunc < self.bestObjFunc):
                self.bestServices = p
                self.bestViolate = violate
                self.bestViolateConstraints = violateConstraints
                self.bestObjFunc = objFunc
                self.stop = 0

            objFunc += violate
            fitness.append(math.exp(-objFunc))
        return np.array(fitness)

    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.popSize), size=self.popSize, replace=True, p=fitness/fitness.sum())
        _pop = [pop[_idx] for _idx in idx]
        return _pop

    def crossoverAndMutation(self, pop):
        newPop = []
        for father in pop:
            child = father.copy()
            if np.random.rand() < self.crossoverRate:
                mother = pop[np.random.randint(self.popSize)]
                cross_points = np.random.randint(0, len(self.services))
                child[cross_points:] = mother[cross_points:].copy()
            newChild = self.mutation(child)
            newPop.append(newChild)
        return newPop

    def mutation(self, child):
        for mutatePoint in self.mutaPointList:
            if np.random.rand() < self.mutationRate:
                rand = np.random.randint(len(self.services[mutatePoint]))
                child[mutatePoint] = self.services[mutatePoint][rand]
        return child

    def start(self):
        pop = []
        for i in range(self.popSize):
            p = []
            for j in self.services:
                rand = np.random.randint(len(j))
                p.append(j[rand])
            pop.append(p)

        for n in range(self.nGenerations):
            pop = self.crossoverAndMutation(pop)
            fitness = self.getFitness(pop, n)
            pop = self.select(pop, fitness)
            self.stop += 1
            if self.stop > self.stopEnd:
                break
        return self.bestServices, self.bestViolate, self.bestObjFunc, self.bestViolateConstraints


def check(PriS, serviceFeatures, constraints, ser2idxdiv, ser2idxmod):
    for s in PriS:
        serIdx = str(ser2idxdiv[s] + 1)
        serCost = serviceFeatures[serIdx][ser2idxmod[s]][-2]
        serQuality = serviceFeatures[serIdx][ser2idxmod[s]][-1]
        serIdx = int(serIdx)

        if not (constraints[serIdx][0] <= serCost <= constraints[serIdx][1] and constraints[serIdx][2] <= serQuality <=
                constraints[serIdx][3]):
            return False
    return True


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


class DPKSD:
    def __init__(self, dataset, reduct, mineFreq, popSize, stop):
        self.dataset = dataset + "/"
        self.reduct = reduct
        self.mineFreq = mineFreq
        self.popSize = popSize
        self.stop = stop

    def start(self):
        servicepattern = mine.mine(self.dataset, self.mineFreq)

        with open(f"./data/{self.dataset}nodefeatures.data", "r") as f:
            nodefeatures = json.load(f)
        with open(f"./data/{self.dataset}labels.data", "r") as f:
            labels = json.load(f)
        with open(f"./data/{self.dataset}serviceFeature.data", "r") as f:
            serviceFeature = json.load(f)
        with open(f"./data/{self.dataset}minCostList.data", "r") as f:
            minCostList = json.load(f)

        serCategory = len(serviceFeature.keys())
        ser2idxdiv = []
        ser2idxmod = []
        for key in serviceFeature.keys():
            index = int(key) - 1
            ser2idxdiv += [index] * len(serviceFeature[key])
            ser2idxmod += [i for i in range(len(serviceFeature[key]))]
        sp2idx = []
        for pattern in servicepattern:
            sp2idx.append([ser2idxdiv[s] for s in pattern])

        newServiceFeatures = []
        constraintsList = []
        for nodefeature, label in zip(nodefeatures[3000:], labels[3000:]):
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
            serviceIndexSet = set(serviceIndex)
            rp2idx = []
            for pattern, idx in zip(servicepattern, sp2idx):
                temp = True
                for sc in idx:
                    if sc not in serviceIndexSet:
                        temp = False
                        break
                if not temp or not check(pattern, serviceFeature, constraints, ser2idxdiv, ser2idxmod):
                    continue
                rp2idx.append(idx)
                for sc in idx:
                    serviceIndexSet.remove(sc)
                if len(serviceIndexSet) <= 1:
                    break
            for idx in serviceIndexSet:
                rp2idx.append([idx])

            spList = [[] for _ in range(len(rp2idx))]
            rp2idxSet = set([tuple(x) for x in rp2idx])
            for pattern, idx in zip(servicepattern, sp2idx):
                if tuple(idx) in rp2idxSet:
                    spList[rp2idx.index(idx)].append(pattern)
            pris = []
            for idx in rp2idxSet:
                if len(idx) == 1:
                    left = ser2idxdiv.index(idx[0])
                    if idx[0] == serCategory - 1:
                        right = len(label)
                    else:
                        right = ser2idxdiv.index(idx[0] + 1)
                    pris += [i for i in range(left, right)]
            pris = addS(pris, serviceFeature, constraints, serviceIndex, ser2idxdiv, ser2idxmod, reduct=self.reduct)
            prisp = []
            for sps in spList:
                if len(sps) > 0:
                    for sp in sps:
                        prisp += [x for x in sp]
            prisp = list(set(prisp))
            prisp = addS(prisp, serviceFeature, constraints, serviceIndex, ser2idxdiv, ser2idxmod, reduct=False)

            _newServiceFeature = []
            for s, sp in zip(pris, prisp):
                if len(s) > 0:
                    _newServiceFeature.append(s)
                else:
                    _newServiceFeature.append(sp)
            newServiceFeatures.append(_newServiceFeature)

            constraint = [[] for _ in range(2)]
            for key, value in constraints.items():
                constraint[0].append(value[-4: -2])
                constraint[1].append(value[-2:])
                break
            constraintsList.append(constraint)

        qualities = {
            "quality": [],
            "time": [],
            "averageQ": 0,
            "averageT": 0
        }
        url = f"./solutions/WOA/{self.dataset}/DPKSD.txt"

        qos = []
        times = []
        idx = 0
        for serviceFeature, constraint, minCost in zip(newServiceFeatures, constraintsList, minCostList[3000:]):
            t = time.time()
            model = GA(constraint, serviceFeature, self.popSize, self.stop)
            bestServices, _, bestObjFunc, _ = model.start()
            qos.append(minCost / bestObjFunc)
            times.append(time.time() - t)
            qualities["quality"] = qos
            qualities["time"] = times
            qualities["averageQ"] = np.average(qos)
            qualities["averageT"] = np.average(times)
            print(idx, np.average(qos), np.average(times))
            idx += 1

        with open(url, "w") as f:
            json.dump(qualities, f)
