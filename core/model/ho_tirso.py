import numpy as np
from numpy.linalg import norm
import copy
from scipy.sparse import dok_array
import pdb



class RFHORSO:
    # _adjacencies
    # _elements
    # _algorithmParam: dict
    # _N
    _uIdx = [[0,2], [0,1], [1,2]]
    _trIdx = [0, 1, 1]
    _luCheck = ["l", "u"]
    _toCheck = ["D", "P", "transform", "featurestd", "expType", "consistency", "gamma", "mu", "at", "lambda"]
 

    def __init__(self, adjacencies, algorithmParam: dict, N: dict):

       
        assert adjacencies.shape[1] == 2 and adjacencies.ndim == 3
        assert len(N) == 2 and self.assertionCheckDict(N, self._luCheck)
        assert self.assertionCheckDict(algorithmParam, self._toCheck)

        self._adjacencies = adjacencies
        self._N = N
        # lambda[i,j]: i = lower or upper, j = numTrans (if consistency is not invoked, ow 1)
        self._maxN = self._adjacencies.shape[-1]
        
        numTrans = 1 + int(algorithmParam["transform"])
        selfN = self._adjacencies.shape[0]
        maxN = max([self._maxN, selfN])

        D = algorithmParam["D"]
        P = algorithmParam["P"]
      
        self._idxList = [2 * D, numTrans, selfN, maxN, 2, P]
        phiIdx = 2 * D * numTrans * maxN * 2 * P
        self._phiNum = [2 * D, numTrans, maxN, 2, P]
        self._algorithmParam = algorithmParam
        self._algorithmParam["lambda"][0,0] = 10
        self._elements = dok_array((phiIdx, selfN))
    
        self._z = dok_array((phiIdx, selfN))
        self._r = dok_array((phiIdx, selfN))
        self._phi = []
        for _ in range(selfN):
            self._phi.append(np.empty(shape = 0))
        self._P = P
        self._D = D
        self.D_indices = np.arange(0, self._D)
        self._selfN = selfN
        self._phiIdx = phiIdx
    
        self._randomFeatures = dict()
        self._randomFreq = dict()
        # self._randomFreq["s"] = np.square(self._algorithmParam["featurestd"]["s"]) * np.random.randn(D, self._selfN, P)
        
        if "initialData" in self._algorithmParam.keys():
            self._luCheck = [x for x in self._luCheck if x in self._algorithmParam["initialData"].keys()]

        if "l" not in self._algorithmParam["initialData"].keys():
            self._uIdx.pop(1)
            self._trIdx.pop(1)

        self.sampleFreq()
        if "initialData" not in self._algorithmParam.keys():
            self._randomFeatures["s"] = np.zeros(shape = [self._selfN, P])
        else:
            self._randomFeatures["s"] = self._algorithmParam["initialData"]["s"]

        if self._algorithmParam["transform"]:
            for x in self._luCheck:
                if "initialData" not in self._algorithmParam.keys():
                    if x == "s":
                        self._randomFeatures[x] = np.zeros(shape = [self._selfN, self._P])
                    else:
                        self._randomFeatures[x] = np.zeros(shape = [self._N[x], self._P])
                else:
                    self._randomFeatures[x] = self._algorithmParam["initialData"][x]
        

        
    def updateParameters(self, incomingData, oneStepData = None):
        self.updateZ(incomingData = incomingData, oneStepData = oneStepData)
        self.updateB()

    def predictData(self):
        return np.squeeze((self._elements * self._z).sum(axis = 0))
    
    def predictDatanStep(self, n):

        allData = np.zeros(shape = [self._selfN, n])
        tempSelf = copy.deepcopy(self)
        
        for idx in range(n):
            allData[:,idx] = tempSelf.predictData()
            tempSelf.updateParameters(allData[:,idx])
        
        return np.squeeze(allData[:,-1])
    
    def extractUpperAdjacency(self, threshold):
        ## TODO: Adjust this case to the new case

        Ns = self._adjacencies.shape[0]
        b = np.zeros(shape = (self._adjacencies.shape[-1], self._P))
        cond  = "u" in self._randomFeatures.keys()
        for p in range(self._P):
            for n in range(Ns):
                uAdjacents = self._adjacencies[n,1,:].astype(np.bool_)
                uIdx = np.nonzero(uAdjacents)[0]

                for tau in uIdx:
                    ix = np.empty(shape = (5,0), dtype = np.int64)

                    tauAdjacents = self._adjacencies[:,1,tau].astype(np.bool_)
                    tauAdjacents = np.nonzero(tauAdjacents)[0]

                    ix = np.append(ix, np.array(np.meshgrid(self.D_indices, 0, tauAdjacents, 1, p)).reshape(5,-1).astype(np.int64), axis = 1)
                    ix = np.append(ix, np.array(np.meshgrid(self.D_indices + self._D, 0, tauAdjacents, 1, p)).reshape(5,-1).astype(np.int64), axis = 1)

                    if cond:
                        ix = np.append(ix, np.array(np.meshgrid(self.D_indices, 1, tau, 1, p)).reshape(5,-1).astype(np.int64), axis = 1)
                        ix = np.append(ix, np.array(np.meshgrid(self.D_indices + self._D, 1, tau, 1, p)).reshape(5,-1).astype(np.int64), axis = 1)
                    
                    flat_indices = np.ravel_multi_index(ix, self._phiNum)
                    
                    b[tau,p] += np.sum(self._elements[flat_indices,[n]]**2)
    
            maxnormb = np.max(b[:,p])
            if maxnormb != 0:
                b[:,p] /= maxnormb
        b = np.sqrt(b)                
              
        b = np.abs(b) > threshold

        return b
        
            
    def sampleFreq(self):
        self._randomFreq["s"] =  np.zeros(shape = (self._D, 2 * self._selfN, self._P))
        self._randomFreq["s"][:,0:self._selfN,:] = self._algorithmParam["featurestd"]["sl"] * np.random.randn(self._D, self._selfN, self._P)
        self._randomFreq["s"][:,self._selfN:2*self._selfN,:] = self._algorithmParam["featurestd"]["su"] * np.random.randn(self._D, self._selfN, self._P)
        
        if self._algorithmParam["transform"]:
            for x in self._luCheck:
                self._randomFreq[x] = self._algorithmParam["featurestd"][x] * np.random.randn(self._D, self._N[x], self._P)


    

    def updateZ(self, incomingData, oneStepData):
        P = self._P
        D = self._D

        gamma = self._algorithmParam["gamma"]
        mu = self._algorithmParam["mu"]

       
        self._grad = dok_array((self._phiIdx, self._selfN))
        self._randomMultiply = dict()
        for x in self._randomFeatures.keys():
            if x == "s":
                self._randomMultiply[x] = np.zeros(shape=(self._D, 2 * self._selfN, self._P))
                self._randomMultiply[x][:,0:self._selfN,:] = self._randomFeatures[x] * self._randomFreq[x][:,0:self._selfN, :]
                self._randomMultiply[x][:,self._selfN:2*self._selfN,:] = self._randomFeatures[x] * self._randomFreq[x][:,self._selfN:2*self._selfN,:]
            else: 
                self._randomMultiply[x] = self._randomFeatures[x] * self._randomFreq[x]

        

        for n in range(self._selfN):
            i = 0
            whole_indices = np.array([])
            for x in self._randomFeatures.keys():
                uIdx = self._uIdx[i]
                trIdx = self._trIdx[i]

                for u in range(uIdx[0], uIdx[1]):
         
                    nuAdjacents = self._adjacencies[n, u, :].astype(np.bool_)
                    if i == 0:
            
                        nAdjacents = self._adjacencies[:, u, nuAdjacents].astype(np.bool_)

                        nAdjacents = np.any(nAdjacents, axis = 1)
                        idxAdjacents = np.nonzero(nAdjacents)[0]
                        numAdjacent = idxAdjacents.shape[0]
                        tempz = np.zeros(shape = [D, numAdjacent, P])
                       
                        for j in range(numAdjacent):
                            idx = idxAdjacents[j]
                            tempz[:,j,:] = self.implementSelfTempZ(x, idx, u, j)
                        nuAdjacents = nAdjacents

                        
                    else:
                        tempz = self.implementTransformTempZ(x, nuAdjacents[0 : self._N[x]], u, j)

                    tempz = tempz.flatten()
                    
                    nuAdjacents_indices = np.nonzero(nuAdjacents)[0]
                    
                   

                    ix = np.array(np.meshgrid(self.D_indices, trIdx, nuAdjacents_indices, u, np.arange(self._idxList[-1]))).reshape(5,-1)

                    
                    
                    flat_indices = np.ravel_multi_index(ix, self._phiNum)
                    curr_indices = flat_indices
                    whole_indices = np.append(whole_indices, flat_indices)
                    self._z[flat_indices,[n]] =  1/np.sqrt(D) * np.cos(tempz)
                   
 
                    ix[0,:] += self._D
                    flat_indices = np.ravel_multi_index(ix, self._phiNum)
                    self._z[flat_indices,[n]] = 1/np.sqrt(D) * np.sin(tempz)
                    
                    whole_indices = np.append(whole_indices, flat_indices)
                    curr_indices = np.append(curr_indices, flat_indices)
                    
                    self._r[curr_indices, [n]] = gamma * self._r[curr_indices, [n]] + mu * incomingData["s"][n] * self._z[curr_indices, [n]]
                    

                i += 1


            if self._phi[n].size == 0:
                self._phi[n] = np.zeros(shape = (whole_indices.size, whole_indices.size))

            self._phi[n] = gamma * self._phi[n] + mu * self._z[whole_indices, [n]].tocsr().transpose() @ self._z[whole_indices, [n]].tocsr()
 
            self._grad[whole_indices, [n]] = self._elements[whole_indices, [n]] @ self._phi[n].transpose() - self._r[whole_indices, [n]]

        for x in self._randomFeatures.keys():
            self._randomFeatures[x][:, 1 : P] = self._randomFeatures[x][:, 0 : P - 1]
            if oneStepData == None:
                self._randomFeatures[x][:, 0] = incomingData[x]
            else:
                self._randomFeatures[x][:,0] = oneStepData[x]
        

    def updateB(self):
        x = self._algorithmParam["expType"][0]
        y = self._algorithmParam["expType"][1]
        numDim = [0]
        if self._algorithmParam["consistency"]:
            numDim.append(1)
        at = self._algorithmParam["at"].send(self._phi)
        next(self._algorithmParam["at"])
        
        if self._algorithmParam["consistency"]:
            loopTr = 1
            trList = [np.arange(self._idxList[1])]
        else:
            loopTr = self._idxList[1]
            trList = [x for x in range(loopTr)]

        differ = self._elements - at * self._grad

        for p in range(self._idxList[-1]):
            for u in range(2):
                lu = ["l", "u"]
                for tau in range(self._N[lu[u]]):
                    nuAdjacents = self._adjacencies[:, u, tau]
                    nIndices = np.nonzero(nuAdjacents)[0]
                    for trIdx in range(loopTr):
                        lmbd = self._algorithmParam["lambda"][u,trIdx]
                        ix = np.array(np.meshgrid(self.D_indices, trList[trIdx], tau, u, p)).reshape(5,-1)
                        ix = np.append(ix, np.array(np.meshgrid(self.D_indices + self._D, trList[trIdx], tau, u, p)).reshape(5,-1), axis = 1)
                        flat_indices = np.ravel_multi_index(ix, self._phiNum)
                        

                        if self.checkExpType(x,y,bool(u)):
                                A = self.thresholdingVectorSparse(differ[flat_indices] [:,nIndices], lmbd, at)

                                for i in range(len(nIndices)):
                                    self._elements[flat_indices, [nIndices[i]]] = A[:,[i]].transpose()

                        else:
                            for n in nIndices:
                                self._elements[flat_indices, [n]] = self.thresholdingVectorSparse(differ[flat_indices, [n]], lmbd, at)

                                
                
    def implementSelfTempZ(self, x, nAdjacents, u, *args):
        out = self._randomMultiply[x][:,u * self._selfN : (u + 1) * self._selfN, :]
        return out[:, nAdjacents, :]
    def implementTransformTempZ(self, x, nuAdjacents, *args):
        return self._randomMultiply[x][:, nuAdjacents, :]
    def reset_at(self):
        self._algorithmParam["at"].send("reset")
        next(self._algorithmParam["at"])
        
    @staticmethod
    def thresholdingVectorSparse(x, lmbd, at):
        normVal = norm(x.data)
        threshold = 1 - at * lmbd/(normVal + 1e-6)
        return (threshold > 0) * threshold * (x)
    
    @staticmethod
    def checkExpType(x, y, i):
        return (not x) and not ((y and i) or not (y or i))
        
    
    @staticmethod
    def assertionCheckDict(d, keyList):
        return all([x in d.keys() for x in keyList])

 