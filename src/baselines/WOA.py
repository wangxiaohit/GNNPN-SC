from src.loadData import loadDataOther, loadDataPN
import numpy as np
import json
import time
import math


class ESWOA:
    def __init__(self, services, constraints, solution=None, popSize=100, MAX_Iter=500):
        self.pe = 0.2
        self.bestFitnesses = []

        if solution is not None:
            for i in range(len(services)):
                for j in range(len(services[i])):
                    services[i][j] = list(services[i][j])
                    services[i][j][0] = round(services[i][j][0], 5)
                    services[i][j][1] = round(services[i][j][1], 5)
                    services[i][j][2] = round(services[i][j][2], 5)
                    services[i][j][3] = round(services[i][j][3], 5)
                    services[i][j] = tuple(services[i][j])
            for i in range(len(solution)):
                solution[i][0] = round(solution[i][0], 5)
                solution[i][1] = round(solution[i][1], 5)
                solution[i][2] = round(solution[i][2], 5)
                solution[i][3] = round(solution[i][3], 5)
                # normal
                if solution[i] == [0.05314, 0.55528, 0.94008, 0.95495]:
                    solution[i][1] = 0.55527
                if solution[i] == [0.03922, 0.56097, 0.94131, 0.92804]:
                    solution[i][1] = 0.56096
                if solution[i] == [0.17292, 0.5995, 0.92651, 0.92459]:
                    solution[i][2] = 0.92652
                if solution[i] == [0.33474, 0.55123, 0.90018, 0.97161]:
                    solution[i][3] = 0.9716
                if solution[i] == [0.73066, 0.40995, 0.90016, 0.92941]:
                    solution[i][3] = 0.92942

                # qws
                if solution[i] == [0.16904, 0.60902, 0.93639, 0.97272]:
                    solution[i][2] = 0.9364
        self.services = services
        self.constraints = constraints
        self.popSize = popSize
        self.qosNum = 4
        self.MAX_Iter = MAX_Iter
        self.consNum = 2

        # initial
        self.pops = []
        for i in range(self.popSize):
            self.pops.append([np.random.choice(list(range(len(service)))) for service in self.services])
        self.popServices = []

        if solution is not None:
            violate, objFunc, _ = self.calc(solution)
            self.bestFitness = violate + objFunc
            self.bestSolutions = solution
            self.bestPops = []
            for l in range(len(solution)):
                service1 = self.services[l]
                service2 = solution[l]
                try:
                    self.bestPops.append(service1.index(tuple(service2)))
                except:
                    self.services[l].append(tuple(service2))
                    service1 = self.services[l]
                    self.bestPops.append(service1.index(tuple(service2)))
            self.initFitness = self.bestFitness
        else:
            self.bestFitness = 3
            self.bestSolutions = None
            self.bestPops = None
            self.initFitness = 3
        self.initPops = self.bestPops

        for i in range(self.popSize):
            service = [self.services[j][self.pops[i][j]] for j in range(len(self.pops[i]))]
            self.popServices.append(service)
            violate, objFunc, _ = self.calc(service)
            fitness = violate + objFunc
            if self.bestFitness > fitness:
                self.bestFitness = fitness
                self.bestSolutions = service
                self.bestPops = self.pops[i]

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

    def start(self):
        t = 0
        while t < self.MAX_Iter:
            prob = 0.2 * (1 - t / self.MAX_Iter)
            # global
            for i in range(self.popSize):
                q = np.random.random()
                if q < prob:
                    rand = np.random.randint(0, len(self.services))
                    randi = np.random.choice(list(range(len(self.services[rand]))))
                    self.pops[i][rand] = randi
                    self.popServices[i][rand] = self.services[rand][randi]
                    violate, objFunc, _ = self.calc(self.popServices[i])
                    fitness = violate + objFunc
                    if self.bestFitness > fitness:
                        self.bestFitness = fitness
                        self.bestSolutions = self.popServices[i]
                        self.bestPops = self.pops[i]

            rand = np.random.random()
            if self.pe > rand:
                t += 1
                self.bestFitnesses.append(self.bestFitness)
                continue
            # local
            for i in range(self.popSize):
                a = 2 - (2 * t / self.MAX_Iter)
                r = np.random.random()
                A = 2 * a * r - a
                C = 2 * r
                l = np.random.random()
                p = np.random.random()
                D = [C * idx1 - idx2 for idx1, idx2 in zip(self.bestPops, self.pops[i])]
                pop_ = None
                if p < 0.5:
                    if abs(A) < 1:
                        pop_ = [round(idx1 - A * idx2) for idx1, idx2 in zip(self.bestPops, D)]
                else:
                    D_ = [idx2 - idx1 for idx1, idx2 in zip(self.bestPops, self.pops[i])]
                    pop_ = [round(idx2 * math.exp(l) * math.cos(2 * math.pi * l) + idx1) for idx1, idx2 in
                            zip(self.bestPops, D_)]
                if pop_ is not None:
                    for j in range(len(pop_)):
                        if abs(pop_[j]) >= len(self.services[j]):
                            pop_[j] %= len(self.services[j])
                    self.pops[i] = pop_
                    self.popServices[i] = [self.services[j][pop_[j]] for j in range(len(pop_))]
                    violate, objFunc, _ = self.calc(self.popServices[i])
                    fitness = violate + objFunc
                    if self.bestFitness > fitness:
                        self.bestFitness = fitness
                        self.bestSolutions = self.popServices[i]
                        self.bestPops = self.pops[i]
            t += 1
            self.bestFitnesses.append(self.bestFitness)
        return self.bestFitness, self.bestSolutions


class WOA:
    def __init__(self, dataset, serCategory, MLESWOAtest, ML2PNWOATest, MLWOATest, ESWOAtest, serviceNumber, reduct,
                 epoch, MAX_Iter, popSize):
        self.dataset = dataset + "/"
        self.serCategory = serCategory
        self.MLESWOAtest = MLESWOAtest
        self.ML2PNWOATest = ML2PNWOATest
        self.MLWOATest = MLWOATest
        self.ESWOAtest = ESWOAtest
        self.serviceNumber = serviceNumber
        self.reduct = reduct
        self.epoch = epoch
        self.MAX_Iter = MAX_Iter
        self.popSize = popSize

        self.times = 0
        self.qosNum = 4
        self.train = False
        self.sSetList = None

    def start(self):
        if self.ML2PNWOATest:
            if self.epoch >= 0:
                with open(f"./solutions/PNHigh/{self.dataset}/allActions{self.epoch}.txt") as f:
                    allActions = json.load(f)
            else:
                with open(f"./solutions/pretrained/{self.dataset[:-1]}-PNHigh.txt") as f:
                    allActions = json.load(f)

            allActionsSolution = [[0] * self.serCategory for _ in range(1000)]
            for i in range(len(allActions)):
                for j in range(len(allActions[i])):
                    allActionsSolution[j][i] = allActions[i][j][: self.qosNum]

            newSolution = []
            self.sSetList = [set() for _ in range(len(allActionsSolution))]
            for i in range(len(allActionsSolution)):
                _newSolution = []
                for action in allActionsSolution[i]:
                    if sum(action[:]) != 3:
                        _newSolution.append(action)
                        _allAction = tuple([round(action[q], 5) for q in range(self.qosNum)])
                        self.sSetList[i].add(_allAction)
                newSolution.append(_newSolution)
        elif self.MLWOATest:
            newSolution = []
            newServiceFeatures, newlabels = loadDataPN(self.epoch, dataset=self.dataset[:-1], serviceNumber=1)
            self.sSetList = [set() for _ in range(len(newServiceFeatures) // 4)]
            idx = 0
            for serviceFeatures in newServiceFeatures[len(newServiceFeatures) // 4 * 3:]:
                _newSolution = []
                for i in range(0, len(serviceFeatures)):
                    if sum(serviceFeatures[i][1: self.qosNum + 1]) != 3:
                        _newSolution.append(serviceFeatures[i][1: self.qosNum + 1])
                        _allAction = tuple([round(serviceFeatures[i][1 + q], 5) for q in range(self.qosNum)])
                        self.sSetList[idx].add(_allAction)
                newSolution.append(_newSolution)
                idx += 1

        else:
            if not self.train:
                newSolution = [None] * 1000
            else:
                newSolution = [None] * 4000

        newServiceFeatures, constraintsList, minCostList = loadDataOther(self.dataset, self.reduct,
                                                                         sSetList=self.sSetList, train=self.train)
        qualitiesInit = {
            "quality": [],
            "time": [],
            "averageQ": 0,
            "averageT": 0
        }

        # ML+ESWOA test
        if self.MLESWOAtest:
            newServiceFeatures, _ = loadDataPN(epoch=self.epoch, dataset=self.dataset[:-1], serviceNumber=self.serviceNumber)  # normal 2 qws 4
            serviceFeatures = []
            serviceCategories = []
            for k in range(len(newServiceFeatures)):
                serviceCategory = []
                serviceFeature = []
                for i in range(len(newServiceFeatures[k]) // self.serviceNumber):
                    _serviceFeature = []
                    for j in range(self.serviceNumber):
                        feature = newServiceFeatures[k][i * self.serviceNumber + j][1: self.qosNum + 1]
                        if sum(feature[1:]) != 3:
                            _serviceFeature.append(tuple(feature))
                    if len(_serviceFeature) > 0:
                        serviceFeature.append(_serviceFeature)
                        serviceCategory.append(i)
                serviceCategory = set(serviceCategory)
                serviceCategories.append(serviceCategory)
                serviceFeatures.append(serviceFeature)
            if self.train:
                newServiceFeatures = serviceFeatures
            else:
                newServiceFeatures = serviceFeatures[len(minCostList) // 4 * 3:]

        bestFitnesses = [[] for _ in range(self.MAX_Iter)]
        if self.train:
            _min = 0
        else:
            _min = len(minCostList) // 4 * 3

        for newServiceFeature, constraints, minCost, solution, idx in zip(newServiceFeatures, constraintsList,
                                                                          minCostList[_min:], newSolution,
                                                                          range(_min, len(minCostList))):
            t = time.time()
            if not solution:
                model = ESWOA(newServiceFeature, constraints, popSize=self.popSize, MAX_Iter=self.MAX_Iter)
            else:
                model = ESWOA(newServiceFeature, constraints, solution, popSize=self.popSize, MAX_Iter=self.MAX_Iter)
            q, sol = model.start()

            for i in range(self.MAX_Iter):
                bestFitnesses[i].append(model.bestFitnesses[i])

            tt = time.time() - t
            qualitiesInit["quality"].append(minCost / q)
            qualitiesInit["time"].append(tt)
            qualitiesInit["averageQ"] = sum(qualitiesInit["quality"]) / (self.times + 1)
            qualitiesInit["averageT"] = sum(qualitiesInit["time"]) / (self.times + 1)
            print(idx, qualitiesInit["averageQ"], qualitiesInit["averageT"])
            self.times += 1

        if self.ML2PNWOATest:
            with open(f"./solutions/WOA/{self.dataset}/ML+2PN+WOA.txt", "w") as f:
                json.dump(qualitiesInit, f)

        if self.ESWOAtest:
            url = f"./solutions/WOA/{self.dataset}/ESWOA.txt"
            with open(url, "w") as f:
                json.dump(qualitiesInit, f)

        if self.MLESWOAtest:
            url = f"./solutions/WOA/{self.dataset}/ML+ESWOA.txt"
            with open(url, "w") as f:
                json.dump(qualitiesInit, f)
