import importlib
import json
import os

import numpy as np

config = json.load(open("configs/deepLogMF.json"))


class Runner:
    def __init__(self, config):
        self.config = config
        self.epochs = config["epochs"]
        Algorithm_Class = importlib.import_module(
            "algorithm."+config["algorithm"]["name"])
        self.algorithm = Algorithm_Class.Algorithm(config["algorithm"])
        self._data_preprocess(
            config["algorithm"]["N"], config["algorithm"]["M"])

        self.train_trace = []

    def _data_preprocess(self, users_num, items_num):
        # if data haven't been pre-processed (convert into matrices)
        if not os.path.exists("np_dataset"):
            os.makedirs("np_dataset")
        if not os.path.exists("np_dataset/dataR.dat"):
            dataset = np.loadtxt("dataset/ratings.dat").astype(np.int32)
            users, items, ratings = dataset[:, 0], dataset[:, 1], dataset[:, 2]
            eval_idx = np.random.choice(
                len(users),
                int(round(0.1*len(users))),
                replace=False,
            )
            dataR = np.zeros([users_num, items_num])
            evalR = np.zeros([users_num, items_num])
            for idx in range(len(users)):
                if idx in eval_idx:
                    evalR[users[idx]-1, items[idx]-1] = 1
                else:
                    dataR[users[idx]-1, items[idx]-1] = 1
            np.savetxt("np_dataset/dataR.dat", dataR)
            np.savetxt("np_dataset/evalR.dat", evalR)

        # load the pre-processed data (as matrices)
        self.dataR = np.loadtxt("np_dataset/dataR.dat")
        self.evalR = np.loadtxt("np_dataset/evalR.dat")

    def train(self):
        for step in range(self.epochs):
            # train
            self.algorithm.train(self.dataR)
            mpr = self.algorithm.eval(self.evalR)
            self.train_trace.append(mpr)
            print("MPR@epoch{}: {:.8}".format(str(step+1).zfill(3), mpr))
        self.save()

    def save(self):
        if not os.path.exists("checkpoint"):
            os.makedirs("checkpoint")
        if not os.path.exists("checkpoint/"+self.config["algorithm"]["name"]):
            os.makedirs("checkpoint/"+self.config["algorithm"]["name"])
        self.algorithm.save("checkpoint/"+self.config["algorithm"]["name"])
        np.savetxt(
            os.path.join(
                "checkpoint/"+self.config["algorithm"]["name"], "trace.txt"),
            np.array(self.train_trace, dtype=np.float32)
        )


if __name__ == "__main__":
    runner = Runner(config)
    runner.train()
