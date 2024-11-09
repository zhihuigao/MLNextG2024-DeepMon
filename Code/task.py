import math
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from Code.loss import LossClassify, LossRegress
from Code.data import DataReader
from Code.Model.model_CNN import CNNNet
from Code.Library.eval import EvalClassify, EvalRegress
from Code.Library.result import ResultClassify, ResultRegress



class GetSignalCsi():
    def __init__(self, time, subList):
        super(GetSignalCsi, self).__init__()
        
        self.time = time
        self.subList = subList

    def GetSignal(self, signalAll):
        time = self.time
        subList = self.subList

        startIdx = (np.shape(signalAll)[0]-time) // 2
        endIdx = (np.shape(signalAll)[0]+time) // 2
        signal = signalAll[startIdx: endIdx][subList]

        return signal



def GetLabelPost(self, labelDict):
    prop = self.prop
    label = labelDict[prop]
    return [label]

def GetLabelPeop(self, labelDict):
    prop = self.prop
    label = labelDict[prop]
    return [label]

def GetLabelDist(self, labelDict):
    prop = self.prop
    label = labelDict[prop]
    return [label]



class Task():
    def __init__(self, opt):
        super(Task, self).__init__()

        DATA_PATH = opt.dataPath
        MODEL_PATH = opt.modelPath
        MODEL_TYPE = opt.modelType

        EPOCH = opt.epoch
        BATCH = opt.batch
        DEVICE = opt.device

        TASK_IN = opt.taskIn
        TASK_OUT = opt.taskOut

        TIME = opt.time
        SUB = [int(i) for i in opt.sub.split(",")]

        POST = [str(i) for i in opt.post.split(",")]
        PEOP = [str(i) for i in opt.peop.split(",")]

        TOKEN = opt.token



        # Basic Setup
        device = torch.device('cuda' if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
        epoch = EPOCH
        batch = BATCH
        
        dataPath = '../' + DATA_PATH + '/'
        if not os.path.exists(dataPath):
            print("Warning: Data NOT Found!")
        modelPath = '../' + MODEL_PATH + '/' + \
            MODEL_TYPE + '_' + str(TASK_IN) + '_' + str(TASK_OUT) + '_' + TOKEN + '/'
        if not os.path.exists(modelPath):
            os.mkdir(modelPath)
        modelType = MODEL_TYPE

        self.device = device
        self.epoch = epoch
        self.batch = batch
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.modelType = modelType



        # Wave Pre-Processing
        if TASK_IN == 'heatmap':
            getSignalCsi = GetSignalCsi(time=TIME, subList=SUB)
            GetSignalIn = getSignalCsi.GetSignal

            timeLen = TIME
            subLen = len(SUB)
        else:
            print("Warning: The wave task NOT found!")



        # Label Pre-Processing
        if TASK_OUT == 'posture':
            GetLabel = GetLabelPost
            outputMode = 'classify'
        elif TASK_OUT == 'people':
            GetLabel = GetLabelPeop
            outputMode = 'classify'
        elif TASK_OUT == 'distance':
            GetLabel = GetLabelDist
            outputLen = 1
            outputMode = 'regression'
        else:
            print("Warning: The output task NOT found!")



        # Prepare Data
        dataFunList = [(GetSignalIn, GetLabel)]

        trainSet = DataReader(dataPath=dataPath, phase='Train', dataFunList=dataFunList, outputMode=outputMode,
            postSet=POST, peopSet=PEOP)
        trainLoader = DataLoader(dataset=trainSet, num_workers=4, batch_size=batch, shuffle=True)
        outputLen = trainSet.GetInfo(modelPath+"trainInfo.txt")
        
        testSet = DataReader(dataPath=dataPath, phase='Test', dataFunList=dataFunList, outputMode=outputMode,
            postSet=POST, peopSet=PEOP)
        testLoader = DataLoader(dataset=testSet, num_workers=4, batch_size=1, shuffle=False)
        outputLenTemp = testSet.GetInfo(modelPath+"testInfo.txt")
        if outputLenTemp != outputLen:
            print("Warning: Train and Test Sets NOT Matched!")

        self.trainLoader = trainLoader
        self.testLoader = testLoader



        # Prepare Model
        if MODEL_TYPE == 'CNN':
            classifierNet = CNNNet(timeLen=timeLen, subLen=subLen, outputLen=outputLen, device=device)
            optimizer = optim.Adam(classifierNet.parameters(), lr=1e-3, weight_decay=2e-5)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=(0, EPOCH, 10), gamma=0.5)
        else:
            print("Warning: Model Type Not Found!")

        if outputMode == 'classify':
            classifierLoss = LossClassify()
            classifierEval = EvalClassify()
            result = ResultClassify
            resultInitParam = (outputLen)
        elif outputMode == 'regress':
            classifierLoss = LossRegress()
            classifierEval = EvalRegress()
            result = ResultRegress
            resultInitParam = None
        else:
            print("Warning: Output Mode NOT Found!")

        self.classifierNet = classifierNet
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.classifierLoss = classifierLoss
        self.classifierEval = classifierEval
        self.result = result
        self.resultInitParam = resultInitParam