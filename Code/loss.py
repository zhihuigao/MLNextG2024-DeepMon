import numpy as np
import torch
import torch.nn as nn



class ClassifierLoss(nn.Module):
    def __init__(self, taskNum, classNum):
        super(ClassifierLoss, self).__init__()

        self.taskNum = taskNum
        self.classNum = classNum
        self.CrossEntropy = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        taskNum = self.taskNum
        classNum = self.classNum
        
        predMat = torch.reshape(pred, (-1, taskNum, classNum))
        loss = 0
        for taskIdx in range(taskNum):
            loss += self.CrossEntropy(predMat[:, taskIdx, :], label[:, taskIdx]) / taskNum
        return loss



class ClassifierEval(nn.Module):
    def __init__(self, taskNum, classNum):
        super(ClassifierEval, self).__init__()

        self.taskNum = taskNum
        self.classNum = classNum

    def forward(self, pred, label):
        taskNum = self.taskNum
        classNum = self.classNum
        
        predMat = torch.reshape(pred, (-1, taskNum, classNum))
        predMat = predMat.cpu().detach().numpy()
        labelMat = label.cpu().detach().numpy()
        batchNum = np.shape(labelMat)[0]

        pos = np.zeros((taskNum))
        mat = np.zeros((taskNum, classNum, classNum))
        predMax = np.zeros((batchNum, taskNum), dtype=int)
        for taskIdx in range(taskNum):
            for batchIdx in range(batchNum):
                predMax[batchIdx, taskIdx] = np.argmax(predMat[batchIdx, taskIdx, :])
                mat[taskIdx, predMax[batchIdx, taskIdx], labelMat[batchIdx, taskIdx]] += 1
            pos[taskIdx] = np.trace(mat[taskIdx, :, :])
                
        evalDict = {
            'pos': pos,
            'mat': mat}
        return evalDict, predMax, labelMat



class RegressionLoss(nn.Module):
    def __init__(self, taskNum, scale=1):
        super(RegressionLoss, self).__init__()

        self.taskNum = taskNum
        self.scale = scale
        self.MSELoss = nn.MSELoss()

    def forward(self, pred, label):
        taskNum = self.taskNum
        scale = self.scale
        
        predMat = torch.reshape(pred, (-1, taskNum))
        loss = 0
        for taskIdx in range(taskNum):
            loss += self.MSELoss(predMat[:, taskIdx], label[:, taskIdx]/scale) / taskNum
        return loss



class RegressionEval(nn.Module):
    def __init__(self, scale=1):
        super(RegressionEval, self).__init__()

        self.scale = scale

    def forward(self, pred, label):
        scale = self.scale
        
        predMat = pred.cpu().detach().numpy() * scale
        labelMat = label.cpu().detach().numpy()

        error = np.abs(predMat - labelMat)
                
        evalDict = {
            'error': error}
        return evalDict, predMat, labelMat