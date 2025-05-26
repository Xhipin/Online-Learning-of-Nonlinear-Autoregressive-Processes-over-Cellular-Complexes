import argparse
from dataset import exampleFunctions, exampleConstantGt, cellularDatasetCreator, cellularDataset
from dataset import functionDataBase, classDataBase
from ..model import RFHORSO
import numpy as np
from os import path, mkdir
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import subprocess, time
import pickle
from tqdm import tqdm


class RFHORSOExp:
    numObj = 0
    _error_keys = ["tvNMSE", "pd", "pfa"]

    """
    =========================
    Initializers
    =========================
    """
    def __init__(self):
        assert RFHORSOExp.numObj == 0, "RFHORSOExp and its subclasses should be called once!!"
        RFHORSOExp.parser = argparse.ArgumentParser()
        self.parser_setup()
        self.argumentParse()
        self._prediction = 0
        self._errors = dict()
        self._message = ""
       
        RFHORSOExp.numObj += 1

    def load(self):
        self._rfhorso = RFHORSO(adjacencies = self._adjacencies, algorithmParam = self._algorithmParam, N = self._N)

    """
    =========================
    Parser
    =========================
    """
    def parser_setup(self):
        RFHORSOExp.parser.add_argument("--datasetName", "-data", default = "synthetic_RFHORSO", type = str)
        RFHORSOExp.parser.add_argument("--plotName", "-pN", default = "data_plot", type = str)
        RFHORSOExp.parser.add_argument("--featureNum", "-D", default = 20, type = int)
        RFHORSOExp.parser.add_argument("--featureDeviationSelfUpper", "-stdsu", default = 1, type = float)
        RFHORSOExp.parser.add_argument("--featureDeviationSelfLower", "-stdsl", default = 1, type = float)
        RFHORSOExp.parser.add_argument("--featureDeviationLower", "-stdl", default = 1, type = float)
        RFHORSOExp.parser.add_argument("--featureDeviationUpper", "-stdu", default = 1, type = float)
        RFHORSOExp.parser.add_argument("--timeLag", "-P", default = 4, type = int)
        RFHORSOExp.parser.add_argument("--transform", "-tr", action='store_false', help="Set transform to False")
        RFHORSOExp.parser.add_argument("--consistency", "-c", default = 1, type = bool)
        RFHORSOExp.parser.add_argument("--experimentType", "-type", default = 1, type = int)
        RFHORSOExp.parser.add_argument("--forgettingFactor", "-gamma", default = 0.98, type = float)
        RFHORSOExp.parser.add_argument("--atFunctionPlace", "-at", default = 0, type = int)
        RFHORSOExp.parser.add_argument("--lambdaLowerSelf", "-lambdals", default = 0.01, type = float)
        RFHORSOExp.parser.add_argument("--lambdaUpperSelf", "-lambdaus", default = 0.01, type = float)
        RFHORSOExp.parser.add_argument("--lambdaLowerTransform", "-lambdaltr", default = 0.01, type = float)
        RFHORSOExp.parser.add_argument("--lambdaUpperTransform", "-lambdautr", default = 0.01, type = float)
        RFHORSOExp.parser.add_argument("--plotting", "-plot", default = 1, type = bool)
        RFHORSOExp.parser.add_argument("--algorithmRun", "-ar", default = 1, type = bool)
        RFHORSOExp.parser.add_argument("--resamplingPeriod", "-period", default = 1000, type = int)
        RFHORSOExp.parser.add_argument("--save", "-s", action = 'store_false', help = 'Use this not to store the data')
        RFHORSOExp.parser.add_argument("--saveExpType", "-set", default = "estimation", type = str)
        RFHORSOExp.parser.add_argument("--futureEstimation", "-Tstep", default=1, type=int)
        RFHORSOExp.parser.add_argument("--earlyStoppingTime", "-est", default = -1, type = int)

        self.additional_parser()
        
    def argumentParse(self):
        
        parsedParameters = RFHORSOExp.parser.parse_args()
        
        if not hasattr(parsedParameters, "rocThreshold"):
            parsedParameters.rocThreshold = 0

        datasetDir = parsedParameters.datasetName
        # Add project_root calculation
        project_root = path.abspath(path.join(path.dirname(__file__), "../../"))
        adjacencyName = "adjacencies.mat"
        self._inputDir = path.join(project_root, "dataset", "Input", datasetDir)
        adjacencyDir = path.join(self._inputDir, adjacencyName)
        atFunctionsDir = path.join(self._inputDir, "at_functions.py")
        if not hasattr(parsedParameters, 'nonlinearityType'):
            parsedParameters.nonlinearityType = - 1
        if not hasattr(parsedParameters, 'dropoutEdges'):
            parsedParameters.dropoutEdges = -1
        self._outputDir = path.join(project_root, "dataset", "Output", datasetDir, f"xp_{parsedParameters.saveExpType}_lambda_{parsedParameters.lambdaUpperSelf}_D_{parsedParameters.featureNum}_nlType_{parsedParameters.nonlinearityType}_std_{parsedParameters.featureDeviationSelfLower}_{parsedParameters.featureDeviationSelfUpper}_de_{parsedParameters.dropoutEdges}_Tstep_{parsedParameters.futureEstimation}_thresh_{parsedParameters.rocThreshold}")
        
        self._plotDir = path.join(self._outputDir, parsedParameters.plotName)
        self._logDir = path.join(self._outputDir, "changes_log.txt")

        assert path.isdir(self._inputDir)
        assert path.isfile(adjacencyDir) and path.isfile(atFunctionsDir)

        mat = loadmat(adjacencyDir)
        
        B1 = mat['B1'].astype(np.bool_).T
        B2 = mat['B2'].astype(np.bool_)

        self._adjacencies = np.zeros(shape = [B1.shape[0], 2, max([B1.shape[1], B2.shape[1]])])
       
        self._adjacencies[:,0,0:B1.shape[1]] = B1
        self._adjacencies[:,1,0:B2.shape[1]] = B2

        sys.path.append(self._inputDir)
        from at_functions import functionDatabase

        self._algorithmParam = dict()

        self._algorithmParam["D"] = parsedParameters.featureNum
        self._algorithmParam["P"] = parsedParameters.timeLag
        self._algorithmParam["transform"] = parsedParameters.transform
        self._algorithmParam["featurestd"] = {"sl": parsedParameters.featureDeviationSelfLower,
                                              "su": parsedParameters.featureDeviationSelfUpper,
                                              "l": parsedParameters.featureDeviationLower, 
                                              "u": parsedParameters.featureDeviationUpper}
        self._algorithmParam["expType"]= self.expTypeMapper(parsedParameters.experimentType)
        self._algorithmParam["consistency"] = parsedParameters.consistency
        self._algorithmParam["gamma"] = parsedParameters.forgettingFactor
        self._algorithmParam["mu"] = 1 - self._algorithmParam["gamma"]
        self._algorithmParam["at"] = functionDatabase[parsedParameters.atFunctionPlace]()
        next(self._algorithmParam["at"])
        self._algorithmParam["lambda"] = np.zeros(shape = [2,2])
        self._algorithmParam["lambda"][0,0] = parsedParameters.lambdaLowerSelf
        self._algorithmParam["lambda"][1,0] = parsedParameters.lambdaUpperSelf
        self._algorithmParam["lambda"][0,1] = parsedParameters.lambdaLowerTransform
        self._algorithmParam["lambda"][1,1] = parsedParameters.lambdaUpperTransform

        self._Tstep = parsedParameters.futureEstimation


        self._N = {"l": B1.shape[-1], "u": B2.shape[-1]}
        self._plotting = parsedParameters.plotting
        self._algRun = parsedParameters.algorithmRun

        self._parsedParameters = parsedParameters

        if not path.isdir(self._outputDir):
            mkdir(self._outputDir)
        self._earlyTime = self._parsedParameters.earlyStoppingTime

        self.initializeData()
      
        self.log_message()
        self.log_git_diff()
        self.log_git_status()

    """
    =========================
    Experimentation
    =========================
    """
    def exp_run(self):

        if not hasattr(self, '_n'):
            self._n = 1
        if self._earlyTime == -2:
            self._earlyTime = int(self._timestep * 0.1)
            self._earlyTime = self._timestep - self._earlyTime
        assert self._earlyTime <= self._timestep, f"Early stopping time should be less than or equal to the timestep: {self._timestep}"
        tqdm.write(f"==== Number of Experiments : {self._n} ====")
        i = 0
        with tqdm(total=self._n, desc="") as pbar:
            for i in range(self._n):
                pbar.set_description(f"Exp:{i + 1}")

                for key in self._error_keys:
                    self._errors[key + "single"].fill(0)

                self.initializeData()
                self.load()
                self.exp()
                pbar.update(1)
                self.saveData()

                single_errors = {k: v for k, v in self._errors.items() if k.endswith("single")}
                errorPathIter = path.join(self._outputDir, f"errors_{i}.pkl")
                with open(errorPathIter, 'wb') as f:
                    pickle.dump(single_errors, f)
        


        # Remove keys ending with "single" after averaging and before plotting/saving
        for key in list(self._errors.keys()):
            if key.endswith("single"):
                del self._errors[key]

        for x in self._errors.keys():
            self._errors[x] /= self._n

        if self._plotting:
            self.plotter()

        if self._parsedParameters.save:
            print("Saving experiment errors and parameters...")
            self.save()
    
      
    def exp(self):
        i = 0
        period = self._parsedParameters.resamplingPeriod
        assert hasattr(self, '_cellData'), "A data should be specified!!!"
        dataMemory = []
       
        with tqdm(total=self._timestep, desc="") as pbar:
            for data in self._cellData:

                if self._algRun:
                    tqdm.write("++++ Iteration Number: " + str(i) + " Lambda: " + str(self._parsedParameters.lambdaUpperSelf)+" ++++")
                    
                    if self._Tstep == 1:
                        if self._earlyTime == -1 or self._earlyTime > i:
                            self._rfhorso.updateParameters(data)
                    elif self._Tstep > 1:
                        if i < self._Tstep - 1:
                            dataMemory.append(data)
                            i += 1
                            continue
                        else:
                            dataMemory.append(data)
                        if self._earlyTime == -1 or self._earlyTime > i:
                            self._rfhorso.updateParameters(incomingData = data, oneStepData = dataMemory[0])
                        dataMemory.pop(0)
                    else:
                        ValueError() # Error
                    self.metrics(data, i)
                    self._prediction = self._rfhorso.predictData()
                
                i += 1
                if i % period == 0:
                    self._rfhorso.sampleFreq()
                    print("Frequencies Resampled")
                if i == self._timestep:
                    break
                pbar.update(1)

    """
    =========================
    Save and Plot
    =========================
    """
    def save(self):
        errorPath = path.join(self._outputDir, "errors.pkl")
        paramPath = path.join(self._outputDir, "parameters.pkl")

        with open(errorPath, 'wb') as file:
            pickle.dump(self._errors, file)

        print(f"Errors are saved to {errorPath}")

        with open(paramPath, 'wb') as file:
            pickle.dump(self._parsedParameters, file)
        print(f"Parameters are saved to {errorPath}")
        

    def saveData(self):
        if hasattr(self, "_cellData"):
            print("Saving Synthetic Data")
            self._cellData.saveData()

    def plotter(self):
        for key in self._error_keys:
            # Create a new figure for each error key
            fig = plt.figure()
            plt.title(key + " error through iterations")
            plt.plot(self._errors[key])

            # Save the figure object to a file using pickle
            with open(f"{self._plotDir}_{key}_figure.pkl", 'wb') as file:
                pickle.dump(fig, file)

            # Close the figure after saving it
            plt.close(fig)

        p10 = int(self._timestep / 10)
        last10p = np.mean(self._errors["tvNMSE"][-p10:])
        print("performance: " + str(last10p))


    """
    =========================
    Metrics
    =========================
    """
    @staticmethod
    def ROC_metric(groundTruth, estimated, wholeAdjacency, P, Nu):
       
        est = estimated[:Nu, :]
        wAd = wholeAdjacency[:,:Nu]
        gt = groundTruth[:,:Nu,:]

        gt = np.all(wAd[:,:,np.newaxis] == gt, axis = 0)
        detections = np.logical_and(gt, est)
        false_alarm = np.logical_and(np.logical_not(gt),est)

        pd = np.sum(detections)/np.sum(gt)
        pfa = np.sum(false_alarm)/np.sum(np.logical_not(gt))

        return pd, pfa

    @staticmethod
    def AUC_metric():
        pass

    @staticmethod
    def EIER_metric():
        pass

    @staticmethod
    def tvNMSE_metric(prediction, groundTruth):
        
        normDiff = ((prediction - groundTruth["s"])**2).mean()
        normgt = (groundTruth["s"]**2).mean()
        errors = normDiff/normgt
        return errors
            
    """
    =========================
    Logging
    =========================
    """
    def log_git_diff(self):
        # Run the Git diff command to capture changes not yet committed
        result = subprocess.run(['git', 'diff'], capture_output=True, text=True)
        log_file = self._logDir
        # If there are changes, log them
        if result.stdout:
            with open(log_file, "a") as log:
                log.write(f"\nChanges detected at {time.ctime()}:\n")
                log.write(result.stdout)  # Write the diff output to the log
        else:
            print("No changes detected.")

    def log_git_status(self):
        # Run the Git status command to see the current status of the repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        log_file = self._logDir
        # Log the current status of the repository
        if result.stdout:
            with open(log_file, "a") as log:
                log.write(f"\nGit status at {time.ctime()}:\n")
                log.write(result.stdout)

    def log_message(self):

        log_file = self._logDir
        
        with open(log_file, "a") as log:
            log.write("\n Additional Message: \n")
            log.write(self._message)

    """
    =========================
    Auxilary
    =========================
    """
    @staticmethod
    def expTypeMapper(expType):
        modthree = expType % 3

        match modthree:
            case 0:
                return [False, False]
            case 1:
                return [False, True]
            case 2:
                return [True, False]
        
    """
    =========================
    Additional Methods
    =========================
    """
    def initializeData(self):
        pass
    def additional_parser(self):
        pass
    def metrics(self, data, i):
        pass
    
    

class RFHORSOSyntheticExp(RFHORSOExp):

    """
    =========================
    Initializer
    =========================
    """
    def __init__(self):
        self._iter = 0
        super().__init__()
        
        for key in self._error_keys:
            self._errors[key] = np.zeros(shape=(self._parsedParameters.totalTime + self._parsedParameters.futureEstimation - 1,))
            self._errors[key + "single"] = np.zeros(shape=(self._parsedParameters.totalTime + self._parsedParameters.futureEstimation - 1,))

    def initializeData(self):
        func = functionDataBase[self._parsedParameters.nonlinearityType]
        clss = classDataBase[self._parsedParameters.exampleFunctionsClassType]
        self._message = f"\n Running with {clss.__class__.__name__} \n"
        self._ex = clss(seed = self._iter, dropoutConnections = self._parsedParameters.dropoutEdges, ul = np.array([self._parsedParameters.dropoutLifting]), func = func, adjacency = self._adjacencies, P = self._parsedParameters.timeLag, dropoutEach = self._parsedParameters.dropoutEach,
                        varl = self._parsedParameters.stdsyntheticlowergaussian , varu = self._parsedParameters.stdsyntheticuppergaussian)
        savePath = path.join(self._inputDir, self._parsedParameters.dataName + str(self._iter) + ".mat")

        if self._parsedParameters.applySeparateVAR:
            self._cellData = cellularDatasetCreator(seed = self._iter, adjacencies = self._adjacencies, P = self._parsedParameters.Psynthetic, nonlinearity = self._ex.nonlinearVAR, N = self._N, savePath = savePath)
        else:
            self._cellData = cellularDatasetCreator(seed = self._iter, adjacencies = self._adjacencies, P = self._parsedParameters.Psynthetic, nonlinearity = self._ex.nonlinearVARCollapsed, N = self._N, savePath = savePath)

        self._algorithmParam["initialData"] = self._cellData.fliplr_data()
        self._timestep = self._parsedParameters.totalTime
        self._n = self._parsedParameters.expNumber
        self._iter += 1
        self._prediction = 0
        self._gtA = self._ex.groundTruthAdjacency
 

    """
    =========================
    Parser
    =========================
    """
    def additional_parser(self):
        RFHORSOExp.parser.add_argument("--nonlinearityType", "-nl", default = 0, type = int)
        RFHORSOExp.parser.add_argument("--dropoutEdges", "-de", default = 0.3, type = float)
        RFHORSOExp.parser.add_argument("--dropoutEach", "-dall", default = 0, type = bool)
        RFHORSOExp.parser.add_argument("--dropoutLifting", "-ul", default = 1, type = bool)
        RFHORSOExp.parser.add_argument("--Psynthetic", "-ps", default = 4, type = int)
        RFHORSOExp.parser.add_argument("--totalTime", "-t", default = 20, type = int)
        RFHORSOExp.parser.add_argument("--dataName", "-name", default = "data", type = str)
        RFHORSOExp.parser.add_argument("--expNumber", "-n", default = 1, type = int)
        RFHORSOExp.parser.add_argument("--rocThreshold", "-threshold", default = 0.5, type = float)
        RFHORSOExp.parser.add_argument("--exampleFunctionsClassType", "-exfunctype", default = 0, type = int)
        RFHORSOExp.parser.add_argument("--stdsyntheticlowergaussian", "-sslg", default = 1, type = float)
        RFHORSOExp.parser.add_argument("--stdsyntheticuppergaussian", "-ssug", default = 1, type = float)
        RFHORSOExp.parser.add_argument("--applySeparateVAR", "-asv", action='store_true', help="Apply separate VAR")
        


    """
    =========================
    Metrics
    =========================
    """
    def metrics(self, data, i):
        b = self._rfhorso.extractUpperAdjacency(self._parsedParameters.rocThreshold)

        pd, pfa = self.ROC_metric(groundTruth=np.squeeze(self._ex.groundTruthAdjacency[:,1,:,:]), estimated=b, wholeAdjacency=np.squeeze(self._adjacencies[:,1,:]), P= self._parsedParameters.timeLag, Nu = self._N['u'])
        temp_error = self.tvNMSE_metric(prediction = self._prediction, groundTruth = data)
 
        self._errors["tvNMSE"][i] += temp_error
        self._errors["tvNMSEsingle"][i] = temp_error
        self._errors["pd"][i] += pd
        self._errors["pdsingle"][i] = pd

        self._errors["pfa"][i] += pfa
        self._errors["pfasingle"][i] = pfa
        tqdm.write("---- tvNMSE: " + str(temp_error) + " ----", end = '\n')
        tqdm.write("---- P_D: " + str(pd) + " ----", end = '\n')
        tqdm.write("---- P_FA: " + str(pfa) + " ----", end = '\n')
        

class RFHORSORealExp(RFHORSOExp):
    _data_keys = ['s', 'l', 'u']
    _error_keys = ["tvNMSE"]

    """
    =========================
    Initializer
    =========================
    """
    def __init__(self):
        self._message = f"\n Running ...\n"
        super().__init__()
       
        dataDir = path.join(self._inputDir, "data.mat")
        assert path.isfile(dataDir)

        data = loadmat(dataDir)
        newData = dict()
        ts = np.inf

        checkS = [x in data.keys() for x in RFHORSORealExp._data_keys]
        if any(checkS[1:]):
            N = self._N
        else:
            N = {}
        for key in RFHORSORealExp._data_keys:
            if key in data.keys():
                newData[key] = data[key].T

                if ts>newData[key].shape[1]:
                    ts = newData[key].shape[1]
            
        self._timestep = ts - self._parsedParameters.timeLag
        data = newData

        for key in self._error_keys:
            self._errors[key] = np.zeros(shape=(self._timestep,))
            self._errors[key + "single"] = np.zeros(shape=(self._timestep,))
        self._algorithmParam["initialData"] = dict()

        for key in data.keys():
            self._algorithmParam["initialData"][key] = np.fliplr(data[key][:,0:self._parsedParameters.timeLag])
        

        self._cellData = cellularDataset(adjacencies = self._adjacencies, N = N, P = self._parsedParameters.timeLag, initialData = data)

    def initializeData(self):
       pass

     

    """
    =========================
    Parser
    =========================
    """
    def additional_parser(self):
        pass

    """
    =========================
    Metrics
    =========================
    """
    def metrics(self, data, i):

        temp_error = self.tvNMSE_metric(prediction = self._prediction, groundTruth = data)
        self._errors["tvNMSE"][i] += temp_error
        self._errors["tvNMSEsingle"][i] += temp_error
        tqdm.write("---- tvNMSE: " + str(temp_error) + " ----", end = '\n')
    



        

    
    