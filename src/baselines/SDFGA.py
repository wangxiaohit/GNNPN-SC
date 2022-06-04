from src.loadData import loadData, loadDataOther
import json
import numpy as np
import time
import gc


class Model:
    def __init__(self, PriS, CorS, SimS, GenS, constraints, nGA, popSize, stop):
        self.GenS = set()
        for i in GenS:
            for s in i:
                self.GenS.add(s)

        self.services = [PriS, CorS, SimS, GenS]
        self.popSize = popSize
        self.qosNum = 4
        self.r = 0.5
        self.stop = stop
        self.bestObjFunc = 0x7777777
        self.bestSolution = None
        self.crossoverRate = 0.5
        self.mutationRate = 0.1
        self.consNum = 2

        self.constraints = [[] for _ in range(self.qosNum)]
        for key, value in constraints.items():
            self.constraints[0].append([1] * len(PriS) + value[-4: -2])
            self.constraints[1].append([1] * len(PriS) + value[-2:])
            break

        self.nGA = [int(round(i * self.popSize)) for i in nGA]
        self.pops = []

        for n in range(len(nGA)):
            rands = np.random.randint(1, 3, self.nGA[n])
            for rand in rands:
                if rand == 1 or n == 3:
                    pop = []
                    for j in range(len(self.services[n])):
                        services = self.services[n][j]
                        if len(services) == 0:
                            services = self.services[0][j] + self.services[1][j] + self.services[2][j] + self.services[3][j]
                        idx = np.random.choice(list(range(len(services))))
                        pop.append(services[idx])
                else:
                    pop = []
                    for j in range(len(self.services[n])):
                        services = self.services[n][j]
                        if len(services) == 0:
                            services = self.services[0][j] + self.services[1][j] + self.services[2][j]
                        if len(services) == 0:
                            services = self.services[0][j] + self.services[1][j] + self.services[2][j] + self.services[3][j]

                        cost = [1 - service[0] for service in services]
                        p = np.array([c / sum(cost) for c in cost])
                        idx = np.random.choice(list(range(len(services))), p=p.ravel())
                        pop.append(services[idx])
                self.pops.append(pop)

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
        if tuple(x[n]) in self.GenS:
            rand = np.random.randint(1, 3)
            if rand == 1 and len(self.services[0][n]) > 0:
                idx = np.random.choice(list(range(len(self.services[0][n]))))
                x[n] = self.services[0][n][idx]
            if rand == 2 and len(self.services[1][n]) > 0:
                idx = np.random.choice(list(range(len(self.services[1][n]))))
                x[n] = self.services[1][n][idx]
            if rand == 3 and len(self.services[2][n]) > 0:
                idx = np.random.choice(list(range(len(self.services[2][n]))))
                x[n] = self.services[2][n][idx]
        else:
            if len(self.services[3][n]) > 0:
                idx = np.random.choice(list(range(len(self.services[3][n]))))
                x[n] = self.services[3][n][idx]
        return x

    def start(self):
        stop = 0
        while stop < self.stop:
            fitness = []
            # selection
            for pop in self.pops:
                violate, objFunc, _ = self.calc(pop)
                if self.bestObjFunc > violate + objFunc:
                    self.bestObjFunc = violate + objFunc
                    self.bestSolution = pop
                    stop = 0
                fitness.append(violate + objFunc)

            _fitness = 3 - np.array(fitness)
            _sum = np.sum(_fitness)
            p = _fitness / _sum
            idxs = np.random.choice(list(range(len(self.pops))), size=round(self.r * self.popSize), replace=False,
                                    p=p.ravel())
            popsNew = [self.pops[idx] for idx in idxs]

            # crossover and mutation
            while len(popsNew) < len(self.pops):
                newChild = []
                idxs = np.random.choice(list(range(len(self.pops))), size=2, replace=False)
                x = self.pops[idxs[0]]
                y = self.pops[idxs[1]]
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
                    if rand < self.crossoverRate:
                        newChild = child
                if len(newChild) > 0:
                    rand = np.random.random()
                    if rand < self.mutationRate:
                        newChild = self.mutate(newChild)
                    popsNew.append(newChild)

            self.pops = popsNew.copy()
            stop += 1
        return self.bestObjFunc


class SDFGA:
    def __init__(self, dataset, reduct, popSize, stop, serCategory):
        self.dataset = dataset + "/"
        self.reduct = reduct
        self.popSize = popSize
        self.stop = stop
        self.serCategory = serCategory

    def addS(self, PriS, serviceFeatures, constraints, serviceIndex, ser2idxdiv, ser2idxmod):
        PriSNew = [[] for _ in range(self.serCategory)]
        for s in PriS:
            serIdx = str(ser2idxdiv[s] + 1)
            ser0 = serviceFeatures[serIdx][ser2idxmod[s]][-4]
            ser1 = serviceFeatures[serIdx][ser2idxmod[s]][-3]
            serCost = serviceFeatures[serIdx][ser2idxmod[s]][-2]
            serQuality = serviceFeatures[serIdx][ser2idxmod[s]][-1]
            serIdx = int(serIdx)
            if constraints[serIdx][0] <= serCost <= constraints[serIdx][1] and constraints[serIdx][2] <= serQuality <= \
                    constraints[serIdx][3]:
                PriSNew[ser2idxdiv[s]].append(tuple([ser0, ser1, serCost, serQuality]))
        _PriSNew = [PriSNew[s] for s in serviceIndex]
        return _PriSNew

    def start(self):
        nodefeatures, serviceFeatureList, _, _, _, labels, _ = loadData(self.dataset)

        x = time.time()
        with open(f"./data/{self.dataset}/minCostList.data") as f:
            minCostList = json.load(f)
        r_all = len(nodefeatures) // 4 * 3
        r_cd = r_all // 6 * 5

        P = 0.5
        K = 0.5
        cb0 = 0
        PriS = set()
        T = sorted(minCostList[:r_all])[r_cd]
        P_cd = r_cd / r_all

        serviceCdTimes = [0] * len(serviceFeatureList)
        serviceTimes = [0] * len(serviceFeatureList)
        for features, label, cost in zip(nodefeatures[:r_all], labels[:r_all], minCostList[:r_all]):
            for l in range(len(label)):
                if label[l] == 1:
                    serviceTimes[l] += 1
                    if cost < T:
                        serviceCdTimes[l] += 1
        p_mscd = [t / r_cd for t in serviceCdTimes]
        p_ms = [t / r_all for t in serviceTimes]
        p_cdms = [round(P_cd * i / j, 3) if j != 0 else 0 for i, j in zip(p_mscd, p_ms)]
        for i in range(len(p_cdms)):
            if p_cdms[i] > P:
                PriS.add(i)
        print(len(PriS))

        del serviceCdTimes
        del serviceTimes
        gc.collect()

        corcd = [[0] * len(serviceFeatureList) for _ in range(len(serviceFeatureList))]
        corcd_ = [[0] * len(serviceFeatureList) for _ in range(len(serviceFeatureList))]
        corcd_cost = dict()

        for label, cost in zip(labels[:r_all], minCostList[:r_all]):
            services = []
            for l in range(len(label)):
                if label[l] == 1:
                    services.append(l)
            for i in range(len(services) - 1):
                for j in range(i + 1, len(services)):
                    if cost < T:
                        corcd[services[i]][services[j]] += 1
                        corcd[services[j]][services[i]] += 1
                    else:
                        corcd_[services[i]][services[j]] += 1
                        corcd_[services[j]][services[i]] += 1
                    if tuple([services[i], services[j]]) not in corcd_cost:
                        corcd_cost[tuple([services[i], services[j]])] = [cost]
                        corcd_cost[tuple([services[j], services[i]])] = [cost]
                    else:
                        corcd_cost[tuple([services[i], services[j]])].append(cost)
                        corcd_cost[tuple([services[j], services[i]])].append(cost)

        cb_cdms = [[(corcd[i][j] - corcd_[i][j]) / r_all if (corcd[i][j] - corcd_[i][j]) / r_all > 0 else 0 for j in range(len(corcd))] for i in range(len(corcd))]

        CorS1 = set()
        serviceCorS1 = set()
        Cor = set()
        for i in range(len(corcd) - 1):
            for j in range(i + 1, len(corcd)):
                if cb_cdms[i][j] > cb0 and corcd[i][j] + corcd_[i][j] > 2:
                    serviceCorS1.add(i)
                    serviceCorS1.add(j)
                    CorS1.add(tuple([i, j]))
                    CorS1.add(tuple([j, i]))
                if corcd[i][j] + corcd_[i][j] > 1:
                    Cor.add(tuple([i, j]))
                    Cor.add(tuple([j, i]))
        PriS -= serviceCorS1

        del corcd
        del corcd_
        del labels
        gc.collect()

        CorS2 = set()
        serviceCorS2 = set()
        for s in PriS:
            F = []
            S = set()
            minF = 1
            maxF = 0
            fList = []
            for i in range(len(serviceFeatureList)):
                if tuple([i, s]) in Cor:
                    f = 0
                    cb = 0
                    cb_ = 0
                    for cost in corcd_cost[tuple([i, s])]:
                        if cost < T:
                            f += 1 - ((serviceFeatureList[i][1] + serviceFeatureList[s][1]) / 2)
                            cb += 1
                        else:
                            f -= 1 - ((serviceFeatureList[i][1] + serviceFeatureList[s][1]) / 2)
                            cb_ += 1
                    if cb > cb_:
                        f = f / (cb - cb_)
                    else:
                        f = 0
                    if f != 0:
                        F.append(tuple([i, s]))
                        fList.append(f)
                        if f > maxF:
                            maxF = f
                        if f < minF:
                            minF = f

            if len(F) >= 2:
                for i in range(len(fList)):
                    fList[i] = (fList[i] - minF) / (maxF - minF)
                fi = np.average(fList)
                if fi <= K:
                    v1 = fList.index(1)
                    if F[v1][0] not in serviceCorS1:
                        serviceCorS2.add(F[v1][0])
                        CorS2.add(tuple([F[v1][0], F[v1][1]]))
                        CorS2.add(tuple([F[v1][1], F[v1][0]]))
                    if F[v1][1] not in serviceCorS1:
                        serviceCorS2.add(F[v1][1])
                        CorS2.add(tuple([F[v1][0], F[v1][1]]))
                        CorS2.add(tuple([F[v1][1], F[v1][0]]))
        del corcd_cost
        gc.collect()

        PriS -= serviceCorS2
        serviceCorS = serviceCorS1 | serviceCorS2
        CorS = CorS1 | CorS2

        del CorS1
        del CorS2
        del serviceCorS1
        del serviceCorS2
        gc.collect()

        GenS = set([i for i in range(len(serviceFeatureList))]) - (PriS | serviceCorS)

        with open(f"./data/{self.dataset}serviceFeature.data", "r") as f:
            serviceFeature = json.load(f)
        ser2idxdiv = []
        ser2idxmod = []
        for key in serviceFeature.keys():
            index = int(key) - 1
            ser2idxdiv += [index] * len(serviceFeature[key])
            ser2idxmod += [i for i in range(len(serviceFeature[key]))]

        SimS = set()
        for k in PriS | serviceCorS:
            l = ser2idxdiv.index(ser2idxdiv[k])
            if ser2idxdiv[k] + 1 >= self.serCategory:
                r = len(serviceFeature)
            else:
                r = ser2idxdiv.index(ser2idxdiv[k] + 1)

            tempS = set([i for i in range(l, r)]) & GenS
            for s in tempS:
                if serviceFeatureList[s][1] < serviceFeatureList[k][1]:
                    SimS.add(s)
        GenS -= SimS

        CorS = serviceCorS.copy()

        with open(f"./data/{self.dataset}/nodefeatures.data", "r") as f:
            nodefeatures = json.load(f)
        with open(f"./data/{self.dataset}/serviceFeature.data", "r") as f:
            serviceFeatures = json.load(f)

        nGA = [len(PriS) / len(serviceFeatureList),
               len(CorS) / len(serviceFeatureList),
               len(SimS) / len(serviceFeatureList),
               len(GenS) / len(serviceFeatureList)]

        times = 0

        url = f"./solutions/WOA/{self.dataset}/SDFGA.txt"
        qualities = {
            "quality": [],
            "time": [],
            "averageQ": 0,
            "averageT": 0
        }

        print(time.time() - x)

        newServiceFeatures, constraintsList, _ = loadDataOther(self.dataset, reduct=self.reduct)
        _min = len(minCostList) // 4 * 3

        for nodefeature, newServiceFeature, woaconstraint, minCost in zip(nodefeatures[_min:],
                                                                          newServiceFeatures,
                                                                          constraintsList, minCostList[_min:]):
            constraints = dict()
            serviceSet = set()
            for i in range(1, self.serCategory + 1):
                constraints[i] = [0] * 8

            for node in nodefeature:
                if node[0] == 1:
                    for i in range(1, self.serCategory + 1):
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

            PriSNew = self.addS(PriS, serviceFeatures, constraints, serviceIndex, ser2idxdiv, ser2idxmod)
            CorSNew = self.addS(CorS, serviceFeatures, constraints, serviceIndex, ser2idxdiv, ser2idxmod)
            SimSNew = self.addS(SimS, serviceFeatures, constraints, serviceIndex, ser2idxdiv, ser2idxmod)
            GenSNew = self.addS(GenS, serviceFeatures, constraints, serviceIndex, ser2idxdiv, ser2idxmod)

            assert len(PriSNew) == len(CorSNew) == len(SimSNew) == len(GenSNew)

            t = time.time()
            model = Model(PriSNew, CorSNew, SimSNew, GenSNew, constraints, nGA, self.popSize, self.stop)
            q = model.start()

            tt = time.time() - t
            qualities["quality"].append(minCost / q)
            qualities["time"].append(tt)
            qualities["averageQ"] = sum(qualities["quality"]) / (times + 1)
            qualities["averageT"] = sum(qualities["time"]) / (times + 1)
            with open(url, "w") as f:
                json.dump(qualities, f)
            print(times, qualities["averageQ"], np.average(qualities["time"]))
            times += 1


