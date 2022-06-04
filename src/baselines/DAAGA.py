from src.loadData import loadDataOther, loadDataPN
import numpy as np
import time
import json


class Model:
    def __init__(self, services, constraints, NGmin, NGmax, NKmax, popSize):
        self.NGmin = NGmin
        self.NGmax = NGmax
        self.NKmax = NKmax
        self._lambda = 0.8
        self.kmax = 60
        self.pc = 0.75
        self.pm = 0.1
        self.popSize = popSize
        self.alpha = 2
        self.beta = 2
        self.pheromoneRange = [100, 200]
        self.rou = 0.4
        self.A = 1
        self.qosNum = 4
        self.consNum = 2
        self.bestObjFunc = 3
        self.bestSolution = None
        self.r = 0.5

        self.services = services
        self.constraints = constraints

        self.tau = []
        for i in range(len(self.services) - 1):
            tau = [[100] * len(self.services[i + 1]) for _ in range(len(self.services[i]))]
            self.tau.append(np.array(tau))

        self.iota = []
        for i in range(len(self.services) - 1):
            iota = [[1 - np.average([self.services[i][j][0], self.services[i + 1][k][0]])
                     for k in range(len(self.services[i + 1]))]
                    for j in range(len(self.services[i]))]
            self.iota.append(np.array(iota))

        self.popServices = []
        for i in range(self.popSize):
            idxs = [np.random.choice(list(range(len(service)))) for service in self.services]
            self.popServices.append([self.services[j][idxs[j]] for j in range(len(idxs))])

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

    def reproduction(self, x, y):
        rand = np.random.randint(1, 3)
        new = []
        newCost = 1

        if rand == 1:
            for i in range(len(x) - 1):
                S0 = x[: i] + y[i:]
                cost = np.average([s[0] for s in S0])
                if cost < newCost:
                    newCost = cost
                    new = S0
        else:
            i = np.random.randint(0, len(x) - 1)
            new = x[: i] + y[i:]
        return new

    def mutate(self, x):
        n = np.random.randint(0, len(x))
        idx = np.random.choice(list(range(len(self.services[n]))))
        x[n] = self.services[n][idx]
        return x

    def mmas(self):
        pops = []
        idxs = np.random.choice(list(range(len(self.services[0]))), self.popSize)
        for i in range(self.popSize):
            pops.append([idxs[i]])
        _pops = []
        for pop in pops:
            _pop = [pop[0]]
            for i in range(len(self.services) - 1):
                now = _pop[i]

                fitness = np.exp(np.multiply(np.array(self.tau[i][now]), self.iota[i][now]))
                _sum = np.sum(fitness)
                p = fitness / _sum
                try:
                    idx = np.random.choice(list(range(len(self.tau[i][now]))), p=p.ravel())
                except:
                    print(self.tau[i][now])
                    print(self.iota[i][now])
                    print(fitness)
                    print(p)
                _pop.append(idx)
            _pops.append(_pop)
        popServices = []
        for i in range(self.popSize):
            service = [self.services[j][_pops[i][j]] for j in range(len(_pops[i]))]
            popServices.append(service)
        return popServices

    def start(self):
        na = round(self._lambda * self.NKmax)

        for nk in range(self.NKmax):
            deltaen_1 = 1
            for ng in range(self.NGmin):
                if nk <= na:
                    fitness = []
                    for pop in self.popServices:
                        violate, objFunc, _ = self.calc(pop)
                        if self.bestObjFunc > violate + objFunc:
                            self.bestSolution = pop
                            self.bestObjFunc = violate + objFunc
                        fitness.append(violate + objFunc)
                    _fitness = 3 - np.array(fitness)
                    _sum = np.sum(_fitness)
                    p = _fitness / _sum
                    idxs = np.random.choice(list(range(len(self.popServices))), size=round(self.r * self.popSize), replace=False,
                                            p=p.ravel())
                    popsNew = [self.popServices[idx] for idx in idxs]

                    while len(popsNew) < len(self.popServices):
                        newChild = []
                        idxs = np.random.choice(list(range(len(self.popServices))), size=2, replace=False)
                        x = self.popServices[idxs[0]]
                        y = self.popServices[idxs[1]]
                        child = self.reproduction(x, y)
                        obx = fitness[idxs[0]]
                        oby = fitness[idxs[1]]

                        violate, objFunc, _ = self.calc(child)
                        obc = violate + objFunc

                        delta = min(obx, oby) - obc
                        if delta > 0:
                            newChild = child
                        else:
                            rand = np.random.random()
                            if rand < self.pc:
                                newChild = child
                        if len(newChild) > 0:
                            rand = np.random.random()
                            if rand < self.pm:
                                newChild = self.mutate(newChild)
                            popsNew.append(newChild)
                else:
                    popsNew = []
                    for pop in self.popServices:
                        rand = np.random.random()
                        newChild = pop.copy()
                        if rand < self.pm:
                            newChild = self.mutate(newChild)
                        popsNew.append(newChild)
                self.popServices = popsNew.copy()

            for ng in range(self.NGmin, self.NGmax):
                fitness = []
                for pop in self.popServices:
                    violate, objFunc, _ = self.calc(pop)
                    if self.bestObjFunc > violate + objFunc:
                        self.bestSolution = pop
                        self.bestObjFunc = violate + objFunc
                    fitness.append(violate + objFunc)
                fitnessMin = np.min(fitness)
                fitnessAvg = np.average(fitness)
                deltaen = fitnessAvg - fitnessMin
                if deltaen < deltaen_1:
                    deltaen_1 = deltaen
                    _fitness = 3 - np.array(fitness)
                    _sum = np.sum(_fitness)
                    p = _fitness / _sum
                    idxs = np.random.choice(list(range(len(self.popServices))), size=round(self.r * self.popSize),
                                            replace=False,
                                            p=p.ravel())
                    popsNew = [self.popServices[idx] for idx in idxs]
                    while len(popsNew) < len(self.popServices):
                        newChild = []
                        idxs = np.random.choice(list(range(len(self.popServices))), size=2, replace=False)
                        x = self.popServices[idxs[0]]
                        y = self.popServices[idxs[1]]
                        child = self.reproduction(x, y)
                        obx = fitness[idxs[0]]
                        oby = fitness[idxs[1]]

                        violate, objFunc, _ = self.calc(child)
                        obc = violate + objFunc

                        delta = min(obx, oby) - obc
                        if delta > 0:
                            newChild = child
                        else:
                            rand = np.random.random()
                            if rand < self.pc:
                                newChild = child
                        if len(newChild) > 0:
                            rand = np.random.random()
                            if rand < self.pm:
                                newChild = self.mutate(newChild)
                            popsNew.append(newChild)
                    self.popServices = popsNew.copy()

                    fitness = []
                    for pop in self.popServices:
                        violate, objFunc, _ = self.calc(pop)
                        if self.bestObjFunc > violate + objFunc:
                            self.bestSolution = pop
                            self.bestObjFunc = violate + objFunc
                        fitness.append(violate + objFunc)
                    _fitness = 3 - np.array(fitness)
                    cqAll = np.sum(_fitness)
                    deltaTau = 1 / cqAll
                    for t in range(len(self.tau)):
                        self.tau[t] = (1 - self.rou) * self.tau[t]
                    for pop in self.popServices:
                        for i in range(len(self.services) - 1):
                            j = self.services[i].index(pop[i])
                            k = self.services[i + 1].index(pop[i + 1])
                            self.tau[i][j][k] = self.tau[i][j][k] + deltaTau
                else:
                    break

            deltaTauBest = 1 / (1 - self.bestObjFunc)
            for t in range(len(self.tau)):
                self.tau[t] = self.rou * self.tau[t]
            for i in range(len(self.services) - 1):
                j = self.services[i].index(self.bestSolution[i])
                k = self.services[i + 1].index(self.bestSolution[i + 1])
                self.tau[i][j][k] = self.tau[i][j][k] + deltaTauBest

            self.popServices = self.mmas()

        return self.bestObjFunc


class DAAGA:
    def __init__(self, dataset, MLESWOAtest, reduct, serviceNumber, epoch, NGmin, NGmax, NKmax, popSize):
        self.dataset = dataset + "/"
        self.MLESWOAtest = MLESWOAtest
        self.reduct = reduct
        self.serviceNumber = serviceNumber
        self.epoch = epoch
        self.NGmin = NGmin
        self.NGmax = NGmax
        self.NKmax = NKmax
        self.popSize = popSize

        self.qosNum = 4

    def start(self):

        times = 0
        if self.MLESWOAtest:
             url = f"./solutions/WOA/{self.dataset}/ML+DAAGA.txt"
        else:
             url = f"./solutions/WOA/{self.dataset}/DAAGA.txt"


        newServiceFeatures, constraintsList, minCostList = loadDataOther(self.dataset, reduct=self.reduct)
        qualities = {
            "quality": [],
            "time": [],
            "averageQ": 0,
            "averageT": 0
        }

        # ML+ESWOA test
        if self.MLESWOAtest:
            newServiceFeatures, _ = loadDataPN(epoch=self.epoch, dataset=self.dataset[:-1], serviceNumber=self.serviceNumber)
            newServiceFeatures = newServiceFeatures[3000:]
            serviceFeatures = []
            for k in range(len(newServiceFeatures)):
                serviceFeature = []
                for i in range(len(newServiceFeatures[k]) // self.serviceNumber):
                    _serviceFeature = []
                    for j in range(self.serviceNumber):
                        feature = newServiceFeatures[k][i * self.serviceNumber + j][1: self.qosNum + 1]
                        if sum(feature[1:]) != 3:
                            _serviceFeature.append(tuple(feature))
                    if len(_serviceFeature) > 0:
                        serviceFeature.append(_serviceFeature)
                serviceFeatures.append(serviceFeature)
            newServiceFeatures = serviceFeatures

        x = time.time()
        for newServiceFeature, constraints, minCost in zip(newServiceFeatures, constraintsList, minCostList[3000:]):
            model = Model(newServiceFeature, constraints, self.NGmin, self.NGmax, self.NKmax, self.popSize)
            t = time.time()
            q = model.start()

            tt = time.time() - t
            qualities["quality"].append(minCost / q)
            qualities["time"].append(tt)
            qualities["averageQ"] = sum(qualities["quality"]) / (times + 1)
            qualities["averageT"] = sum(qualities["time"]) / (times + 1)

            print(times, qualities["averageQ"], qualities["averageT"])
            times += 1

        with open(url, "w") as f:
            json.dump(qualities, f)
