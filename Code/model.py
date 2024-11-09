import torch
import torch.nn as nn



def comp2vec(comp, dim):
    compReal = torch.real(comp).unsqueeze(dim).float()
    compImag = torch.imag(comp).unsqueeze(dim).float()
    vec = torch.cat((compReal, compImag), dim=dim)
    return vec

class ResBlock2D(nn.Module):
    def __init__(self, channelNum,
        kernel_size=[5, 5], padding=[2, 2], stride=[1, 1]):
        super(ResBlock2D, self).__init__()

        self.Layer = nn.Sequential(
            nn.Conv2d(
                in_channels=channelNum, out_channels=channelNum,
                kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=channelNum),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channelNum, out_channels=channelNum, 
                kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=channelNum))

    def forward(self, x_0):
        x_1 = self.Layer(x_0) + x_0
        return x_1

class ResNet(nn.Module):
    def __init__(self, symNum, symLen, outputLen, isSTFT=True):
        super(ResNet, self).__init__()

        self.symNum = symNum
        self.symLen = symLen
        self.outputLen = outputLen
        self.isSTFT = isSTFT

        channelNum = 16
        self.PreConv = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=channelNum,
                kernel_size=[5, 5], padding=[2, 2], stride=[1, 1]),
            nn.BatchNorm2d(num_features=channelNum),
            nn.ReLU(inplace=True))
        
        self.resNum = 6
        ResList = []
        for _ in range(self.resNum):
            ResBlock = ResBlock2D(channelNum=channelNum, 
                kernel_size=[5, 5], padding=[2, 2], stride=[1, 1])
            ResList.append(ResBlock)
        self.ResList = nn.ModuleList(ResList)

        featList = [128, 128, outputLen]
        seluList = [True, True, False]
        self.lineNum = len(featList)
        LineList = []
        featOut = int(symNum*symLen*channelNum)
        for lineIdx in range(self.lineNum):
            featIn = featOut
            featOut = featList[lineIdx]
            if seluList[lineIdx]:
                Line = nn.Sequential(
                    nn.Linear(in_features=featIn, out_features=featOut),
                    nn.BatchNorm1d(num_features=featOut),
                    nn.SELU(inplace=True))
            else:
                Line = nn.Sequential(
                    nn.Linear(in_features=featIn, out_features=featOut))
            LineList.append(Line)
        self.LineList = nn.ModuleList(LineList)

    def forward(self, x_0):
        x_1 = torch.fft.fft(x_0, dim=-1) if self.isSTFT else x_0
        x_1 = comp2vec(x_1, 1)
        
        x_2 = self.PreConv(x_1)
        
        x_3_out = x_2
        for resIdx in range(self.resNum):
            x_3_in = x_3_out
            x_3_out = self.ResList[resIdx](x_3_in)
        x_3 = x_3_out

        x_4_out = torch.flatten(x_2, start_dim=1, end_dim=-1)
        for lineIdx in range(self.lineNum):
            x_4_in = x_4_out
            x_4_out = self.LineList[lineIdx](x_4_in)
        x_4 = x_4_out

        return x_4
