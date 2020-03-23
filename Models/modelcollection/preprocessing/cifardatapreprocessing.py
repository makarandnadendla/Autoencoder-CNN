#limit the data for the classes bird (2), deer (4) and truck (9) to 50%
import numpy as np

def CifarDataPreprocessing(trainX, trainY, indeces, percentRatio = 0.5, randomSeed = 101):
    np.random.seed(randomSeed)
    
    # initialize a drop array 
    dropIndeces = np.array([])
    
    # go through each of the class labels and select a percentage of them randomly
    for index in indeces:
        i, j = np.where(trainY == index)
        sampleSize = np.int(len(i)*percentRatio)
        sample = np.random.choice(i, size = sampleSize, replace = False)
        dropIndeces = np.concatenate([dropIndeces, sample])
    
    # create a boolean mask from the values we want to remove
    dropIndeces = dropIndeces.astype(int)
    mask = np.ones(len(trainY), dtype=bool)
    mask[dropIndeces] = False
    
    # filter the values to be dropped out
    trainX = trainX[mask]
    trainY = trainY[mask]
    return trainX, trainY