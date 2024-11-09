import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os

import sys
sys.path.append('..')
from Code.decoding import *


def Normalize(wave_0, power=1):
    wave_1 = wave_0 - np.mean(wave_0)
    waveNorm = np.sqrt(np.mean(np.abs(wave_1)**2))
    if waveNorm != 0:
        wave_2 = wave_1 / waveNorm * np.sqrt(power)
    else:
        wave_2 = wave_1 * 0
    return wave_2

class DataReader(Dataset):
    def __init__(self, dataPath, rate, isAug=True,
            taskIn=[0, 1, 2, 3, 4, 5, 6], taskOut='prot',
            protSet=['Non-HT', 'HT', 'HTGF', 'VHT', 'others'], symTime=4e-6):
        super(DataReader, self).__init__()

        symList = [int(sym) for sym in taskIn.split(',')]
        symNum = len(symList)
        symLen = int(np.ceil(rate*symTime))

        if taskOut == 'prot':
            taskNum = 1
            classNum = len(protSet)
            mode = 'classifier'
        elif taskOut == 'LSIG':
            taskNum = 24
            classNum = 2
            mode = 'classifier'
        elif taskOut == 'HTSIG':
            taskNum = 48
            classNum = 2
            mode = 'classifier'
        elif taskOut == 'VHTSIGA':
            taskNum = 48
            classNum = 2
            mode = 'classifier'
        elif taskOut == 'time':
            taskNum = 1
            classNum = 1
            mode = 'regression'
        elif taskOut == 'psdu':
            taskNum = 1
            classNum = 1
            mode = 'regression'
        elif taskOut == 'mcs':
            taskNum = 1
            classNum = 10
            mode = 'classifier'
        else:
            print('Warning: Task Out NOT Supported!')

        waveList = []
        labelList = []
        folderList = os.listdir(dataPath)
        for folder in tqdm(folderList):
            folderPath = dataPath + folder + '/'
            if(not os.path.isdir(folderPath))or(folder[0]=='_'):
                continue
            wavePath = folderPath + 'Low/'
            labelPath = folderPath+ 'Label/'
            if(not os.path.exists(wavePath))or(not os.path.exists(labelPath)):
                continue
            
            waveFileList = os.listdir(wavePath)
            for waveFile in waveFileList:
                if not waveFile.endswith('.bin'):
                    continue
                labelFile = waveFile[:-4] + '.mat'
                if not os.path.exists(labelPath+labelFile):
                    continue
                
                waveAll = np.fromfile(open(wavePath+waveFile, 'r'), dtype=np.complex64)
                wave = np.zeros((symNum, symLen), dtype=complex)
                if np.shape(waveAll)[0] < max(symList)*symLen:
                    continue
                for symIdx in range(symNum):
                    sym = symList[symIdx]
                    offset = int(np.floor(sym*symLen))
                    wave[symIdx, :] = Normalize(waveAll[offset: offset+symLen], power=1)

                labelDict = loadmat(labelPath+labelFile)
                prot = labelDict['prot'][0]
                if (not prot in protSet):
                    continue
                if taskOut == 'prot':
                    label = np.array([protSet.index(prot)])
                elif taskOut == 'LSIG':
                    label = np.squeeze(labelDict['LSIGBit'])
                elif taskOut == 'HTSIG':
                    label = np.squeeze(labelDict['HTSIGBit'])
                elif taskOut == 'VHTSIGA':
                    label = np.squeeze(labelDict['VHTSIGABit'])
                elif taskOut == 'time':
                    lsig = np.squeeze(labelDict['LSIGBit'])
                    label = np.array([LSIG2Time(lsig)])
                elif taskOut == 'psdu':
                    if protSet[0] == 'Non-HT':
                        lsig = np.squeeze(labelDict['LSIGBit'])
                        labelTemp = LSIG2PSDU(lsig)
                    elif protSet[0] == 'HT':
                        htsig = np.squeeze(labelDict['HTSIGBit'])
                        labelTemp = HTSIG2PSDU(htsig)
                    else:
                        print('Warning: Protocol NOT Supported!')
                    label = np.array([labelTemp])
                elif taskOut == 'mcs':
                    if protSet[0] == 'Non-HT':
                        lsig = np.squeeze(labelDict['LSIGBit'])
                        labelTemp, _, _ = LSIG2MCS(lsig)
                    elif protSet[0] == 'HT':
                        htsig = np.squeeze(labelDict['HTSIGBit'])
                        labelTemp, _, _ = HTSIG2MCS(htsig)
                    elif protSet[0] == 'VHT':
                        vhtsiga = np.squeeze(labelDict['VHTSIGABit'])
                        labelTemp, _, _ = VHTSIGA2MCS(vhtsiga)
                    else:
                        print('Warning: Protocol NOT Supported!')
                    if np.isnan(labelTemp):
                        continue
                    label = np.array([labelTemp])
                else:
                    print('Warning: Task Out NOT Supported!')
                waveList.append(wave)
                labelList.append(label)

        self.isAug = isAug
        self.symNum = symNum
        self.symLen = symLen
        self.taskNum = taskNum
        self.classNum = classNum
        self.mode = mode
        self.waveList = waveList
        self.labelList = labelList
    
    def GetInfo(self, infoFilePath):
        symNum = self.symNum
        symLen = self.symLen
        taskNum = self.taskNum
        classNum = self.classNum
        mode = self.mode
        labelList = self.labelList
        
        if mode == 'classifier':
            countMat = np.zeros((taskNum, classNum), dtype=int)
            for label in labelList:
                for taskIdx in range(taskNum):
                    countMat[taskIdx, label[taskIdx]] += 1
            
            info = ''
            for taskIdx in range(taskNum):
                info += str(taskIdx) + ': '
                for classIdx in range(classNum):
                    info += str(countMat[taskIdx, classIdx])+', '
                info += '\n'
            infoFile = open(infoFilePath, "w")
            infoFile.write(info)
            infoFile.close()
        
        return symNum, symLen, taskNum, classNum, mode

    def __getitem__(self, index):
        isAug = self.isAug
        mode = self.mode
        wave = self.waveList[index]
        label = self.labelList[index]
        
        if isAug:
            augPhase = np.exp(1j*2*np.pi*np.random.rand()) if isAug else 1
            waveAug = wave * augPhase
        else:
            waveAug = wave
        waveTensor = torch.from_numpy(waveAug)
        if mode == 'classifier':
            labelTensor = torch.from_numpy(np.array(label)).long()
        else:
            labelTensor = torch.from_numpy(np.array(label)).float()
        
        return waveTensor, labelTensor

    def __len__(self):
        return len(self.labelList)
