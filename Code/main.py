import argparse
import numpy as np
import os
from scipy.io import savemat
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append('..')
from Code.loss import ClassifierLoss, ClassifierEval, RegressionLoss, RegressionEval
from Code.data import DataReader
from Code.model import ResNet



parser = argparse.ArgumentParser(description='WiFi Packet Type Classification')
parser.add_argument('--dataPath', default='Data/Zhihui_new', type=str,
    help='data path')
parser.add_argument('--modelPath', default='Model/Zhihui_new', type=str,
    help='model path')
parser.add_argument('--modelType', default='ResNet', type=str, 
    choices=['ResNet'],
    help='neural network model')
parser.add_argument('--epoch', default=100, type=int,
    help='epoch number')
parser.add_argument('--batch', default=64, type=int,
    help='batch size')
parser.add_argument('--device', default='gpu', type=str,
    help='running device')
parser.add_argument('--taskIn', default='0,1,2,3,4,5,6,7,8,9,10,11', type=str, 
    help='classfication input symbol indices, splited by , with no space')
parser.add_argument('--taskOut', default='prot', type=str, 
    choices=[
        'prot', 'LSIG', 'HTSIG', 'VHTSIGA',
        'mcs', 'time', 'psdu'],
    help='classfication output')
parser.add_argument('--scale', default=1, type=float,
    help='loss function scaling factor')
parser.add_argument('--rate', default=2000000, type=int,
    help='input sampling rate')
parser.add_argument('--prot', default='Non-HT,HT,HTGF,VHT,others', type=str,
    help='protocal type (Non-HT, HT, HTGF, VHT, others), splited by , with no space')

parser.add_argument('--token', default='test', type=str,
    help='training token')



if __name__ == '__main__':
    opt = parser.parse_args()
    DATA_PATH = opt.dataPath
    MODEL_PATH = opt.modelPath
    MODEL_TYPE = opt.modelType
    EPOCH = opt.epoch
    BATCH = opt.batch
    DEVICE = opt.device
    TASK_IN = opt.taskIn
    TASK_OUT = opt.taskOut
    SCALE = opt.scale
    RATE = opt.rate
    PROT = opt.prot
    TOKEN = opt.token

    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    
    dataPath = '../' + DATA_PATH + '/SampleRate_' + str(RATE) + '/'
    if not os.path.exists(dataPath):
        print('Warning: Data NOT Found!')
    modelPath = '../' + MODEL_PATH + '/' + \
        MODEL_TYPE + '_' + str(TASK_OUT) + '_' + str(RATE) + '_' + TOKEN + '/'
    if not os.path.exists(modelPath):
        os.mkdir(modelPath)
    else:
        print('Warning: Result Exists!')
        exit()

    protSet = PROT.split(',')
    trainSet = DataReader(dataPath=dataPath+'Train/', rate=RATE, isAug=True, taskIn=TASK_IN, taskOut=TASK_OUT, protSet=protSet)
    trainLoader = DataLoader(dataset=trainSet, num_workers=4, batch_size=BATCH, shuffle=True)
    symNum, symLen, taskNum, classNum, mode = trainSet.GetInfo(modelPath+'train_info.txt')
    testSet = DataReader(dataPath=dataPath+'Test/', rate=RATE, isAug=False, taskIn=TASK_IN, taskOut=TASK_OUT, protSet=protSet)
    testLoader = DataLoader(dataset=testSet, num_workers=4, batch_size=1, shuffle=False)
    testSet.GetInfo(modelPath+'test_info.txt')

    if mode == 'classifier':
        if MODEL_TYPE == 'ResNet':
            Model = ResNet(symNum=symNum, symLen=symLen, outputLen=taskNum*classNum, isSTFT=True)
        else:
            print('Warning: Model Type Not Found!')
        Loss = ClassifierLoss(taskNum=taskNum, classNum=classNum)
        Eval = ClassifierEval(taskNum=taskNum, classNum=classNum)
    else:
        if MODEL_TYPE == 'ResNet':
            Model = ResNet(symNum=symNum, symLen=symLen, outputLen=taskNum, isSTFT=True)
        else:
            print('Warning: Model Type Not Found!')
        Loss = RegressionLoss(taskNum=taskNum, scale=SCALE)
        Eval = RegressionEval(scale=SCALE)
    optimizer = optim.Adam(Model.parameters(), lr=1e-3)
    print('Model parameter number:', sum(param.numel() for param in Model.parameters()))

    Model.to(device)
    Loss.to(device)
    Eval.to(device)

    lossMin = +np.inf
    for epochIdx in range(EPOCH):
        # Train Model
        Model.train()
        trainBar = tqdm(trainLoader)
        trainResult = {
            'dataNum': 0, 'loss': 0,
            'pos': np.zeros((taskNum)), 'mat': np.zeros((taskNum, np.abs(classNum), np.abs(classNum))),
            'error': np.zeros((0, taskNum))}
        for wave, label in trainBar:
            batchNum = label.size(0)
            if batchNum < BATCH:
                continue

            wave = wave.to(device)
            label = label.to(device)

            # Train Network
            optimizer.zero_grad()

            pred = Model(wave)
            loss = Loss(pred, label)
            result, _, _ = Eval(pred, label)

            loss.backward()
            optimizer.step()

            # Save Result
            trainResult['dataNum'] += batchNum
            trainResult['loss'] += batchNum * loss.item()
            if mode == 'classifier':
                trainResult['mat'] += result['mat']
                trainResult['pos'] += result['pos']
                trainBar.set_description(
                    desc='[%d/%d] (Train) loss=%.4f, acc=%.4f'
                        %(epochIdx, EPOCH, 
                            trainResult['loss']/trainResult['dataNum'],
                            np.mean(trainResult['pos']/trainResult['dataNum'])))
            else:
                trainResult['error'] = np.concatenate((trainResult['error'], result['error']), axis=0)
                trainBar.set_description(
                    desc='[%d/%d] (Train) loss=%.4f, error=%.4f'
                        %(epochIdx, EPOCH, 
                            trainResult['loss']/trainResult['dataNum'],
                            np.mean(trainResult['error'])))
        savemat(modelPath + 'train_%d.mat' % (epochIdx), trainResult)

        # Test Model
        Model.eval()
        testBar = tqdm(testLoader)
        testResult = {
            'dataNum': 0, 'loss': 0,
            'pos': np.zeros((taskNum)), 'mat': np.zeros((taskNum, np.abs(classNum), np.abs(classNum))),
            'error': np.zeros((0, taskNum))}
        detail = {
            'predMax': np.zeros((0, taskNum)), 'labelMat': np.zeros((0, taskNum))}
        for wave, label in testBar:
            batchNum = label.size(0)

            wave = wave.to(device)
            label = label.to(device)

            # Test Network
            pred = Model(wave)
            loss = Loss(pred, label)
            result, predMax, labelMat = Eval(pred, label)

            # Save Result
            testResult['dataNum'] += batchNum
            testResult['loss'] += batchNum * loss.item()
            detail['predMax'] = np.concatenate((detail['predMax'], predMax), axis=0)
            detail['labelMat'] = np.concatenate((detail['labelMat'], labelMat), axis=0)
            if mode == 'classifier':
                testResult['mat'] += result['mat']
                testResult['pos'] += result['pos']
                testBar.set_description(
                    desc='[%d/%d] (Test) loss=%.4f, acc=%.4f'
                        %(epochIdx, EPOCH, 
                            testResult['loss']/testResult['dataNum'],
                            np.mean(testResult['pos']/testResult['dataNum'])))
            else:
                testResult['error'] = np.concatenate((testResult['error'], result['error']), axis=0)
                testBar.set_description(
                    desc='[%d/%d] (Test) loss=%.4f, error=%.4f'
                        %(epochIdx, EPOCH, 
                            testResult['loss']/testResult['dataNum'],
                            np.mean(testResult['error'])))
        savemat(modelPath + 'test_%d.mat' % (epochIdx), testResult)
        
        lossNow = testResult['loss']/testResult['dataNum']
        if lossNow < lossMin:
            lossNow = lossMin
            savemat(modelPath + 'detail.mat', detail)

        # Save Model
        # torch.save(Model.state_dict(), modelPath+'net_epoch_%d.pth'%(epochIdx))
