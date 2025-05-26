import numpy as np
from numpy import sin, cos
from scipy.io import savemat

"""
=========================
=========================
DATASET
=========================
=========================
"""

class cellularDatasetCreator:

    """
    =========================
    Initializer
    =========================
    """
    def __init__(self, seed, adjacencies, P, nonlinearity, N = {}, **kwargs):
        lenN = len(N.values())
        assert lenN == 0 or (lenN == 2 and all([x in ["l", "u"] for x in N.keys()]))

        self._adjacencies = adjacencies
        self._P = P
        self._N = N
        self._selfN = self._adjacencies.shape[0]
        self._func = nonlinearity


        if any([x == "initialData" for x in kwargs.keys()]):
            self._data = kwargs["initialData"]
        else:
            rng = np.random.default_rng(seed = seed)
            self._data = dict()
            self._data["s"] = rng.normal(0, 1, size = [self._selfN, self._P])
            for x in self._N.keys():
                self._data[x] = rng.normal(0, 1, size = [self._N[x], self._P])

        if any([x == "savePath" for x in kwargs.keys()]):
            self._savePath = kwargs["savePath"]

    """
    =========================
    Iterators
    =========================
    """
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.datasetIterator()
    
    """
    =========================
    Getters
    =========================
    """
    @property
    def HOTIRSOLastState(self, hotirsoObject):
        return hotirsoObject
    
    @property
    def data(self):
        return self._data

    """
    =========================
    Data Operations
    =========================
    """
    def fliplr_data(self):
        lr_data = dict()
        for x in self._data.keys():
            lr_data[x] = np.fliplr(self._data[x])
        return lr_data
    
    """
    =========================
    Data Saver
    =========================
    """
    def saveData(self):
        if hasattr(self,"_savePath"):
            assert self._savePath.endswith('.mat')
            savemat(file_name = self._savePath, mdict = self._data)
    """
    =========================
    Deconstructor
    =========================
    """
    def __del__(self):
        pass
   
        
    """
    =========================
    Additional Methods
    =========================
    """
    def datasetIterator(self):

        lastData = dict()
        for x in self._data.keys():
            lastData[x] = self._data[x][:, -self._P : ]
        
        newData = self._func(lastData, self._adjacencies, self._P)

        for x in self._data.keys():
            self._data[x] = np.append(self._data[x], newData[x][:,np.newaxis], axis = 1)
        
        return newData

        
    
class cellularDataset(cellularDatasetCreator):
    """
    =========================
    Initializer
    =========================
    """
    def __init__(self, adjacencies, P, initialData, N = {}, **kwargs):
        super().__init__(0, adjacencies, P, None, N, initialData = initialData)
        self._iterator = 0

    """
    =========================
    Iterator
    =========================
    """
    def datasetIterator(self):
        newData = dict()
        for x in self._data.keys():
            newData[x] = self._data[x][:,self._iterator + self._P]
        self._iterator += 1
        return newData
    
"""
=========================
=========================
EXAMPLE FUNCTIONS
=========================
=========================
"""
class exampleFunctions:
    _lu = ["l", "u"]
    """
    =========================
    Initializer
    =========================
    """
    def __init__(self, seed, dropoutConnections, ul, dropoutEach, adjacency, P, func, varl, varu):
        self._dropout = dropoutConnections
        self._rng = np.random.default_rng(seed = seed)
        self._ul = ul
        self._func = np.vectorize(lambda x, v, vl, vu: func(np.array([x]),v, vl, vu))
        self._gtAdjacency = self.dropoutRandom(adjacency, P)
        self._currAdjacency = self._gtAdjacency
        self._de = dropoutEach
        self.create_self_adjacency()
        self._varl = varl
        self._varu = varu

    """
    =========================
    Getters
    =========================
    """

    @property
    def groundTruthAdjacency(self):
        return self._gtAdjacency
    
    """
    =========================
    Function Creator
    =========================
    """
    def nonlinearVAR(self, data, adjacency, P):

        if self._de:
            self._gtAdjacency = self.dropoutRandom(adjacency, P)
        
     
            
        newData = dict()
        newData["s"] = np.zeros(shape = [data["s"].shape[0]])

        
        for u in data.keys():
            if u in self._lu:
                newData[u] = self._rng.normal(0, 0.0, size = data[u].shape[0])
            else:
                for p in range(P):
                    
                    for v in self._lu:
                        currSelfData = self._func(data["s"][:,p], v, self._varl, self._varu)
                        currTransformedData = self._func(data[v][:,p],v, self._varl, self._varu)

                        vIdx = self._lu.index(v)

                        newData[u] += self._self_adjacency[:,vIdx,:,p] @ currSelfData
                        newData[u] += self._gtAdjacency[:,vIdx,:currTransformedData.shape[0],p] @ currTransformedData

                                
        return newData
    
    def nonlinearVARCollapsed(self, data, adjacency, P):

        if self._de:
            self._gtAdjacency = self.dropoutRandom(adjacency, P)
            
        newData = dict()
        newData["s"] = np.zeros(shape = [data["s"].shape[0]])

        collapsed_adjacency = np.logical_or(self._self_adjacency[:, 0, :, :],
                                    self._self_adjacency[:, 1, :, :])
        
        for u in data.keys():
            if u in self._lu:
                newData[u] = self._rng.normal(0, 0.0, size = data[u].shape[0])
            else:
                for p in range(P):
                    
                    for v in data.keys():
                        currTransformedData = self._func(data[v][:,p], v, self._varl, self._varu)
                        if v == "s":
                            newData[u][:] += collapsed_adjacency[:,:,p] @ currTransformedData
                        else:
                            vIdx = self._lu.index(v)
                            newData[u][:] += self._gtAdjacency[:,vIdx,:currTransformedData.shape[0],p] @ currTransformedData
                                
        return newData

        
    
    """
    =========================
    Example Functions
    =========================
    """
    @staticmethod
    def g1(data, *args):
        x = np.mean(data)
        return 0.25 * sin(x**2) + 0.25 * sin(2 * x) + 0.5 * sin(x)

    @staticmethod
    def g2(data, *args):
        x = np.mean(data)
        return 0.25 * cos(x**2) + 0.25 * cos(2 * x) + 0.5 * cos(x)
    
    @staticmethod
    def g3(data, *args):
        x = np.mean(data)
        return (1/np.sqrt(2 * np.pi)) * np.exp(-x**2/2)
    
    @staticmethod
    def g4(data, *args):
        x = np.mean(data)
        return 0.25 * sin(2 * x) + 0.5 * sin(x)
    
    @staticmethod
    def g5(data, *args):
        x = np.mean(data)
        return 0.25 * sin(np.sqrt(np.abs(x))) + 0.25 * sin(2 * x) + 0.5 * sin(x)
    
    @staticmethod
    def g6(data, *args):
        x = np.mean(data)
        return 0.25 * sin(0.2 * x) + 0.5 * sin(0.1*x)
    
    @staticmethod
    def g7(data, *args):
        x = np.mean(data)
        u = args[0]
        varl = args[1]
        varu = args[2]

        if not u:
            return (1/np.sqrt(2 * np.pi * varl)) * np.exp(-x**2/(2 * varl))
        return (1/np.sqrt(2 * np.pi * varu)) * np.exp(-x**2/(2 * varu))

    """
    =========================
    Dropout Mechanism
    =========================
    """
    def dropoutRandom(self, adjacency, P):
        newAdjacency = np.repeat(adjacency[:, :, :, np.newaxis], P, axis=3)
        ul = self._ul.astype(np.int8)
    
        return self.dropper(newAdjacency=newAdjacency, ul=ul)
    
    """
    =========================
    Additional Functions
    =========================
    """
    def dropper(self, newAdjacency, ul):
        toDrop = self._rng.binomial(1, self._dropout, newAdjacency.shape[-2:]).astype(np.bool_)
        newAdjacency[:, ul, toDrop] = 0

        return newAdjacency
    
    def create_self_adjacency(self):
        # Create a zero-initialized self-adjacency matrix
        num_edges = self._gtAdjacency.shape[0]
        P = self._gtAdjacency.shape[3]
        self._self_adjacency = np.zeros((num_edges, 2, num_edges, P), dtype=bool)

        for u in [0, 1]:  # u=0 for B1, u=1 for B2
            for p in range(P):
                for e in range(num_edges):
                    # Find which nodes are adjacent to edge e at time p for mode u
                    adj_vec = self._gtAdjacency[e, u, :, p]
                    # Self-adjacency defined as sharing common nodes in B1/B2
                    for f in range(num_edges):
                        if np.any(self._gtAdjacency[f, u, adj_vec.astype(bool), p]):
                            self._self_adjacency[e, u, f, p] = True

class exampleConstantGt(exampleFunctions):
    """
    =========================
    Initializer
    =========================
    """
    def __init__(self, seed, dropoutConnections, ul, dropoutEach, adjacency, P, func, varl, varu):
        super().__init__(seed, dropoutConnections, ul, dropoutEach, adjacency, P, func, varl, varu)

    """
    =========================
    Dropout Mechanism
    =========================
    """
    def dropper(self, newAdjacency, ul):
        toDrop = self._rng.binomial(1, self._dropout, newAdjacency.shape[-2:-1]).astype(np.bool_)
        newAdjacency[:, ul, toDrop,:] = 0
    
        return newAdjacency
