import src.models.trainML as trainML
import src.models.trainPNLow as trainPNLow
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