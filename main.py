import src.loadData as loadData
import src.models.trainML as trainML
import sys
import configparser


if __name__ == "__main__":
    dataset = sys.argv[1]
    approach = sys.argv[2]
    config = configparser.RawConfigParser()
    config.read("environment.ini")

    if (dataset == "QWS" or dataset == "Normal") and approach == "ML":
        parakey = config.options("QWS-ML")
        paravalue = [config.get("QWS-ML", key) for key in parakey]
        model = trainML.TrainML("QWS", int(paravalue[0]), int(paravalue[1]), int(paravalue[2]), int(paravalue[3]),
                                float(paravalue[4]), float(paravalue[5]), int(paravalue[6]))
        model.start()


