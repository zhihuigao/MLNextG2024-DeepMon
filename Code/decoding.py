import numpy as np



def Dec2Bin(dec, digit=None):
    # LSB: bin[0]
    # MSB: bin[-1]
    digitMax = 100
    decTemp = dec

    binArray = np.zeros((digitMax))
    index = digitMax
    while (decTemp > 0)and(index >= 1):
        binArray[index-1] = decTemp % 2
        decTemp //= 2
        index -= 1

    if digit == None:
        bin = binArray[index+1:]
    else:
        bin = binArray[-digit:]
    bin = np.flip(bin)
    
    return bin

def Bin2Dec(bin):
    # LSB: bin[0]
    # MSB: bin[-1]
    binArray = np.flip(bin)
    binLen = np.shape(binArray)[0]

    dec = 0
    for binIdx in range(binLen):
        dec = dec * 2 + binArray[binIdx]
    return dec



def LSIG2MCS(lsig):
    # only for Non-HT
    mcsBit = lsig[0: 3]
    mcsDec = Bin2Dec(mcsBit)
    if mcsDec == 3:
        mcs = 0
        modu = 2
        code = 1
    elif mcsDec == 7:
        mcs = 1
        modu = 2
        code = 3
    elif mcsDec == 2:
        mcs = 2
        modu = 4
        code = 1
    elif mcsDec == 6:
        mcs = 3
        modu = 4
        code = 3
    elif mcsDec == 1:
        mcs = 4
        modu = 16
        code = 1
    elif mcsDec == 5:
        mcs = 5
        modu = 16
        code = 3
    elif mcsDec == 0:
        mcs = 6
        modu = 64
        code = 2
    elif mcsDec == 4:
        mcs = 7
        modu = 64
        code = 3
    return mcs, modu, code

def LSIG2PSDU(lsig):
    # only for Non-HT
    psduBit = lsig[5: 17]
    psduDec = Bin2Dec(psduBit)
    psdu = psduDec
    return psdu

def LSIG2Time(lsig):
    # only for HT/VHT
    timeBit = lsig[5: 17]
    timeDec = Bin2Dec(timeBit)
    time = (4*np.floor((timeDec+3)/3)+20) * 1e-6
    return time



def HTSIG2MCS(htsig):
    # only for HT
    mcsBit = htsig[0: 3]
    mcsDec = Bin2Dec(mcsBit)
    mcs = int(mcsDec)
    if mcsDec == 0:
        modu = 2
        code = 1
    elif mcsDec == 1:
        modu = 4
        code = 1
    elif mcsDec == 2:
        modu = 4
        code = 3
    elif mcsDec == 3:
        modu = 16
        code = 1
    elif mcsDec == 4:
        modu = 16
        code = 3
    elif mcsDec == 5:
        modu = 64
        code = 2
    elif mcsDec == 6:
        modu = 64
        code = 3
    elif mcsDec == 7:
        modu = 64
        code = 5
    return mcs, modu, code

def HTSIG2Ant(htsig):
    # only for HT
    antBit = htsig[3: 5]
    antDec = Bin2Dec(antBit)
    ant = antDec
    return ant

def HTSIG2Band(htsig):
    # only for HT
    bandBit = htsig[7]
    if bandBit == 0:
        band = 'CBW20'
    else:
        band = 'CBW40'
    return band

def HTSIG2PSDU(htsig):
    # only for HT
    psduBit = htsig[8: 24]
    psduDec = Bin2Dec(psduBit)
    psdu = psduDec
    return psdu



def VHTSIGA2MCS(vhtsiga):
    # only for VHT
    mcsBit = vhtsiga[29: 33]
    mcsDec = Bin2Dec(mcsBit)
    mcs = int(mcsDec)
    if mcsDec == 0:
        modu = 2
        code = 1
    elif mcsDec == 1:
        modu = 4
        code = 1
    elif mcsDec == 2:
        modu = 4
        code = 3
    elif mcsDec == 3:
        modu = 16
        code = 1
    elif mcsDec == 4:
        modu = 16
        code = 3
    elif mcsDec == 5:
        modu = 64
        code = 2
    elif mcsDec == 6:
        modu = 64
        code = 3
    elif mcsDec == 7:
        modu = 64
        code = 5
    elif mcsDec == 8:
        modu = 256
        code = 3
    elif mcsDec == 9:
        modu = 256
        code = 5
    else:
        print('Warning: MCS NOT Found!')
        mcs = np.nan
        modu = np.nan
        code = np.nan
    return mcs, modu, code