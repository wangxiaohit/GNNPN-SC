import src.baselines.DPKSD.fpgrowth as fpgrowth
import time
import json


def mine(dataset, n):
    t = time.time()
    with open(f"./data/{dataset}/labels.data", "r") as f:
        data = json.load(f)
        parsedDat = []
        for d in data[:3000]:
            _parsedDat = []
            for idx in range(len(d)):
                if d[idx] == 1:
                    _parsedDat.append(idx)
            parsedDat.append(_parsedDat)
    initSet = fpgrowth.createInitSet(parsedDat)
    myFPtree, myHeaderTab = fpgrowth.createFPtree(initSet, n)
    freqItems = []
    fpgrowth.mineFPtree(myFPtree, myHeaderTab, n, set([]), freqItems)
    servicePatterns = []
    for x in freqItems:
        if len(x) >= 2:
            servicePatterns.append(x)
    print("Mining Time:", time.time() - t)
    return servicePatterns
