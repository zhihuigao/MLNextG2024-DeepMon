import numpy as np



class ResultClassify():
    def __init__(self, resultInitParam):
        super(ResultClassify, self).__init__()

        outputLen = resultInitParam

        self.result = {
            'dataNum': 0, 
            'loss': 0.0,
            'pos': 0,
            'mat': np.zeros(outputLen, outputLen)}

    def update(self, loss, evalDict, dataNum):
        self.result['dataNum'] += dataNum
        self.result['loss'] += loss * dataNum

        self.result['pos'] += evalDict['pos']
        self.result['mat'] += evalDict['mat']
        
        return None

    def show(self):
        result = self.result

        desc = "Loss: %.4f AccAve: %.4f" %(
            result['loss'] / (result['dataNum']+1e-10),
            np.mean(result['posList']) / (result['dataNumIn']+1e-10))

        return desc

    def output(self):
        return self.result



class ResultRegress():
    def __init__(self, resultInitParam):
        super(ResultRegress, self).__init__()

        self.result = {
            'dataNum': 0, 
            'loss': 0.0, 
            'error': 0.0,
            'itemPred': [], 'itemTarget': []}

    def update(self, loss, evalDict, dataNum):
        self.result['dataNum'] += dataNum
        self.result['loss'] += loss * dataNum

        self.result['error'] += evalDict['error'] * dataNum
        self.result['itemPred'] += evalDict['itemPred']
        self.result['itemTarget'] += evalDict['itemTarget']
        
        return None


    def show(self):
        result = self.result

        desc = "Loss: %.4f Error: %.4f" %(
            result['loss'] / (result['dataNum']+1e-10),
            np.mean(result['error']) / (result['dataNum']+1e-10))

        return desc

    def output(self):
        return self.result
