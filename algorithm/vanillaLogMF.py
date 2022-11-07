import os
import numpy as np


def log_confidence(alpha, epsilon, x):
    return 1 + alpha * np.log(1+x/epsilon)


def linear_confidence(alpha, x):
    return alpha * x


class Algorithm:
    def __init__(self, config):
        self.F = config["F"]
        self.M = config["M"]
        self.N = config["N"]

        self.X = np.random.normal(0, .001, size=[self.N, self.F])
        self.Y = np.random.normal(0, .001, size=[self.M, self.F])
        self.BetaX = np.random.normal(size=[self.N])
        self.BetaY = np.random.normal(size=[self.M])

        self.gradXcnt = np.zeros([self.N, self.F])
        self.gradYcnt = np.zeros([self.M, self.F])
        self.gradBetaXcnt = np.zeros([self.N])
        self.gradBetaYcnt = np.zeros([self.M])

        self.alpha = None
        self.epsilon = config["epsilon"]
        self.lamda = config["lamda"]
        self.gamma = config["gamma"]

        self.confidence_func = globals(
        )[config["confidence_function"]+"_confidence"]

        self.train_step = 0

    def train(self, dataR):
       # compute confidence w.r.t. the chosen function
        if not self.alpha:
            self.alpha = np.sum(dataR == 0) / np.sum(dataR)
        confidence = self.confidence_func(self.alpha, self.epsilon, dataR)

        # compute gradient for X and BetaX
        predR = (np.matmul(self.X, self.Y.T).T + self.BetaX).T + self.BetaY
        expR = np.exp(predR)
        common_factor = (confidence-expR) / (1+expR)
        gradX = np.matmul(common_factor, self.Y) - self.lamda * self.X
        gradBetaX = np.sum(common_factor, axis=1)
        # update X and BetaX
        self.gradXcnt += np.square(gradX)
        self.gradBetaXcnt += np.square(gradBetaX)
        self.X += self.gamma * gradX / np.sqrt(self.gradXcnt)
        self.BetaX += self.gamma * gradBetaX / np.sqrt(self.gradBetaXcnt)

        # compute gradient for Y and BetaY
        predR = (np.matmul(self.X, self.Y.T).T + self.BetaX).T + self.BetaY
        expR = np.exp(predR)
        common_factor = (confidence-expR) / (1+expR)
        gradY = np.matmul(self.X.T, common_factor).T - self.lamda * self.Y
        gradBetaY = np.sum(common_factor, axis=0)
        # update Y and BetaY
        self.gradYcnt += np.square(gradY)
        self.gradBetaYcnt += np.square(gradBetaY)
        self.Y += self.gamma * gradY / np.sqrt(self.gradYcnt)
        self.BetaY += self.gamma * gradBetaY / np.sqrt(self.gradBetaYcnt)

    def eval(self, evalR):
        rankR = np.argsort(
            -((np.matmul(self.X, self.Y.T).T + self.BetaX).T + self.BetaY),
            axis=1,
        ).argsort(axis=1) / (self.M-1)
        return np.sum(evalR*rankR) / np.sum(evalR)

    def save(self, path):
        np.savetxt(os.path.join(path, "X.txt"), self.X)
        np.savetxt(os.path.join(path, "BetaX.txt"), self.BetaX)
        np.savetxt(os.path.join(path, "Y.txt"), self.Y)
        np.savetxt(os.path.join(path, "BetaY.txt"), self.BetaY)

    def load(self, path):
        self.X = np.loadtxt(os.path.join(path, "X.txt"))
        self.BetaX = np.loadtxt(os.path.join(path, "BetaX.txt"))
        self.Y = np.loadtxt(os.path.join(path, "Y.txt"))
        self.BetaY = np.loadtxt(os.path.join(path, "BetaY.txt"))
