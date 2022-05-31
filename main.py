import src.models.trainML as trainML
import src.models.trainPNLow as trainPNLow
import src.models.trainPNHigh as trainPNHigh
import src.baselines.WOA as WOA
import src.baselines.DAAGA as DAAGA
import src.baselines.SDFGA as SDFGA
import sys
import configparser


if __name__ == "__main__":
    dataset = sys.argv[1]
    approach = sys.argv[2]
    config = configparser.RawConfigParser()
    config.read("environment.ini")

    if (dataset == "QWS" or dataset == "qws") and approach == "ML":
        parakey = config.options("QWS-ML")
        paravalue = [config.get("QWS-ML", key) for key in parakey]
        model = trainML.TrainML("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                                float(paravalue[4]), float(paravalue[5]), int(paravalue[6]))
        model.start()

    if dataset == "Normal" and approach == "ML":
        parakey = config.options("Normal-ML")
        paravalue = [config.get("Normal-ML", key) for key in parakey]
        model = trainML.TrainML("Normal", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                                float(paravalue[4]), float(paravalue[5]), int(paravalue[6]))
        model.start()

    if (dataset == "QWS" or dataset == "qws") and approach == "PNLow":
        parakey = config.options("QWS-PNLow")
        paravalue = [config.get("QWS-PNLow", key) for key in parakey]
        model = trainPNLow.PNLow("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                                 int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]),
                                 int(paravalue[8]), float(paravalue[9]), float(paravalue[10]), float(paravalue[11]),
                                 int(paravalue[12]))
        model.start()

    if dataset == "Normal" and approach == "PNLow":
        parakey = config.options("Normal-PNLow")
        paravalue = [config.get("Normal-PNLow", key) for key in parakey]
        model = trainPNLow.PNLow("Normal", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                                 int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]),
                                 int(paravalue[8]), float(paravalue[9]), float(paravalue[10]), float(paravalue[11]),
                                 int(paravalue[12]))
        model.start()

    if (dataset == "QWS" or dataset == "qws") and approach == "PNHigh":
        parakey = config.options("QWS-PNHigh")
        paravalue = [config.get("QWS-PNHigh", key) for key in parakey]
        model = trainPNHigh.PNHigh("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                                   int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]),
                                   int(paravalue[8]), float(paravalue[9]), float(paravalue[10]), float(paravalue[11]),
                                   int(paravalue[12]), int(paravalue[12]))
        model.start()

    if dataset == "Normal" and approach == "PNHigh":
        parakey = config.options("Normal-PNHigh")
        paravalue = [config.get("Normal-PNHigh", key) for key in parakey]
        model = trainPNHigh.PNHigh("Normal", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                                   int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]),
                                   int(paravalue[8]), float(paravalue[9]), float(paravalue[10]), float(paravalue[11]),
                                   int(paravalue[12]), int(paravalue[12]))
        model.start()

    if (dataset == "QWS" or dataset == "qws") and approach == "WOA":
        parakey = config.options("QWS-WOA")
        paravalue = [config.get("QWS-WOA", key) for key in parakey]
        model = WOA.WOA("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                        int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]),
                        int(paravalue[8]), int(paravalue[9]))
        model.start()

    if dataset == "Normal" and approach == "WOA":
        parakey = config.options("Normal-WOA")
        paravalue = [config.get("Normal-WOA", key) for key in parakey]
        model = WOA.WOA("Normal", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                        int(paravalue[4]), int(paravalue[5]), float(paravalue[6]), int(paravalue[7]),
                        int(paravalue[8]), int(paravalue[9]))
        model.start()

    if (dataset == "QWS" or dataset == "qws") and approach == "ML+ESWOA":
        parakey = config.options("QWS-ML+ESWOA")
        paravalue = [config.get("QWS-ML+ESWOA", key) for key in parakey]
        model = WOA.WOA("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                        int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]),
                        int(paravalue[8]), int(paravalue[9]))
        model.start()

    if dataset == "Normal" and approach == "ML+ESWOA":
        parakey = config.options("Normal-ML+ESWOA")
        paravalue = [config.get("Normal-ML+ESWOA", key) for key in parakey]
        model = WOA.WOA("Normal", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                        int(paravalue[4]), int(paravalue[5]), float(paravalue[6]), int(paravalue[7]),
                        int(paravalue[8]), int(paravalue[9]))
        model.start()

    if (dataset == "QWS" or dataset == "qws") and approach == "ESWOA":
        parakey = config.options("QWS-ESWOA")
        paravalue = [config.get("QWS-ESWOA", key) for key in parakey]
        model = WOA.WOA("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                        int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]),
                        int(paravalue[8]), int(paravalue[9]))
        model.start()

    if dataset == "Normal" and approach == "ESWOA":
        parakey = config.options("Normal-ESWOA")
        paravalue = [config.get("Normal-ESWOA", key) for key in parakey]
        model = WOA.WOA("Normal", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                        int(paravalue[4]), int(paravalue[5]), float(paravalue[6]), int(paravalue[7]),
                        int(paravalue[8]), int(paravalue[9]))
        model.start()

    if (dataset == "QWS" or dataset == "qws") and approach == "ML+DAAGA":
        parakey = config.options("QWS-ML+DAAGA")
        paravalue = [config.get("QWS-ML+DAAGA", key) for key in parakey]
        model = DAAGA.DAAGA("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                            int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]))
        model.start()

    if dataset == "Normal" and approach == "ML+DAAGA":
        parakey = config.options("Normal-ML+DAAGA")
        paravalue = [config.get("Normal-ML+DAAGA", key) for key in parakey]
        model = DAAGA.DAAGA("Normal", int(paravalue[0]), float(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                           int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]))
        model.start()

    if (dataset == "QWS" or dataset == "qws") and approach == "DAAGA":
        parakey = config.options("QWS-DAAGA")
        paravalue = [config.get("QWS-DAAGA", key) for key in parakey]
        model = DAAGA.DAAGA("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                            int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]))
        model.start()

    if dataset == "Normal" and approach == "DAAGA":
        parakey = config.options("Normal-DAAGA")
        paravalue = [config.get("Normal-DAAGA", key) for key in parakey]
        model = DAAGA.DAAGA("Normal", int(paravalue[0]), float(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                            int(paravalue[4]), int(paravalue[5]), int(paravalue[6]), int(paravalue[7]))
        model.start()

    if (dataset == "QWS" or dataset == "qws") and approach == "SDFGA":
        parakey = config.options("QWS-SDFGA")
        paravalue = [config.get("QWS-SDFGA", key) for key in parakey]
        model = SDFGA.SDFGA("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]))
        model.start()

    if dataset == "Normal" and approach == "SDFGA":
        parakey = config.options("Normal-SDFGA")
        paravalue = [config.get("Normal-SDFGA", key) for key in parakey]
        model = SDFGA.SDFGA("Normal", float(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]))
        model.start()