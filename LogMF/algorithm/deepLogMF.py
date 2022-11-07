import os
import numpy as np
import torch
from torch import nn


def log_confidence(alpha, epsilon, x):
    return 1 + alpha * torch.log(1+x/epsilon)


def linear_confidence(alpha, x):
    return alpha * x


class Algorithm:
    def __init__(self, config):
        self.F = config["F"]
        self.M = config["M"]
        self.N = config["N"]

        self.X = nn.parameter.Parameter(
            torch.as_tensor(np.random.normal(0, .001, size=[self.N, self.F])),
            requires_grad=True,
        )
        self.Y = nn.parameter.Parameter(
            torch.as_tensor(np.random.normal(0, .001, size=[self.M, self.F])), 
            requires_grad=True,
        )
        self.BetaX = nn.parameter.Parameter(
            torch.as_tensor(np.random.normal(size=[self.N])), 
            requires_grad=True,
        )
        self.BetaY = nn.parameter.Parameter(
            torch.as_tensor(np.random.normal(size=[self.M])), 
            requires_grad=True,
        )

        self.X_optimizer = torch.optim.Adagrad(
            [
                {"params": self.X},
                {"params": self.BetaX},
            ],
            lr=config["gamma"],
        )
        self.Y_optimizer = torch.optim.Adagrad(
            [
                {"params": self.Y},
                {"params": self.BetaY},
            ],
            lr=config["gamma"],
        )

        self.alpha = None
        self.epsilon = config["epsilon"]
        self.lamda = config["lamda"]

        self.confidence_func = globals(
        )[config["confidence_function"]+"_confidence"]

        self.train_step = 0
    
    def train(self, dataR):
        if not self.alpha:
            self.alpha = np.sum(dataR == 0) / np.sum(dataR)
        dataR = torch.as_tensor(dataR)
        confidence = self.confidence_func(self.alpha, self.epsilon, dataR)
        
        # update X and BetaX
        predR = (torch.matmul(self.X, self.Y.transpose(1, 0)).transpose(1, 0) + self.BetaX).transpose(1, 0) + self.BetaY
        obj = torch.mean(
            confidence * predR - \
            (1+confidence) * torch.log(1+torch.exp(predR))
        ) - \
        torch.mean(.5 * self.lamda * torch.square(self.X)) - \
        torch.mean(.5 * self.lamda * torch.square(self.Y))
        self.X_optimizer.zero_grad()
        (-obj).backward()
        self.X_optimizer.step()

        # update Y and BetaY
        predR = (torch.matmul(self.X, self.Y.transpose(1, 0)).transpose(1, 0) + self.BetaX).transpose(1, 0) + self.BetaY
        obj = torch.mean(
            confidence * predR - \
            (1+confidence) * torch.log(1+torch.exp(predR))
        ) - \
        torch.mean(.5 * self.lamda * torch.square(self.X)) - \
        torch.mean(.5 * self.lamda * torch.square(self.Y))
        self.Y_optimizer.zero_grad()
        (-obj).backward()
        self.Y_optimizer.step()

    def eval(self, evalR):
        evalR = torch.as_tensor(evalR)
        rankR = torch.argsort(
            -((torch.matmul(self.X, self.Y.transpose(1, 0)).transpose(1, 0) + self.BetaX).transpose(1, 0) + self.BetaY),
            axis=1,
        ).argsort(axis=1) / (self.M-1)
        return (torch.sum(evalR*rankR) / torch.sum(evalR)).item()
    
    def save(self, path):
        torch.save(self.X, os.path.join(path, "X.pt"))
        torch.save(self.Y, os.path.join(path, "Y.pt"))
        torch.save(self.BetaX, os.path.join(path, "BetaX.pt"))
        torch.save(self.BetaY, os.path.join(path, "BetaY.pt"))
    
    def load(self, path):
        self.X = torch.load(os.path.join(path, "X.pt"))
        self.Y = torch.load(os.path.join(path, "Y.pt"))
        self.BetaX = torch.load(os.path.join(path, "BetaX.pt"))
        self.BetaY = torch.load(os.path.join(path, "BetaY.pt"))
