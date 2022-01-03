### 
# KDTree-Predictor: tracking and predicting cells from a developing embryo
# using the KDTree structure.
# Copyright (C) 2021  Mirela Teixeira Cazzolato <mirelac@usp.br>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
###

##############################################################################################################
# KDTree-Predictor
##############################################################################################################

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as dist
from sklearn.neighbors import KDTree

##############################################################################################################
# Parameters and Data Structure
##############################################################################################################

# Set of cells from the embryo
embryo = pd.DataFrame(columns = ['Cell', 'active'])

# Set of predictors for the cells
predictors = pd.DataFrame(columns = ['id', 'Predictor'])

# Output vector
output = pd.DataFrame(columns = ['id', 'total_cells', 'active_cells'])

# # KDTree-Predictor's parameters:
# w                # Window size
# th               # Threshold parameter (only used for error)
# pw               # Percentage of the window to be discarded
# maxError         # Maximum error allowed for prediction
# r                # Relaxation parameter

# KDTree-Predictor's control variables:
buffer = 0
numberOfPredictions = 0
numberOfNonPredictions = 0
isWindowInitialized = False

##############################################################################################################
# Print embryo information
##############################################################################################################

def printEmbryo():
    for i in range(len(embryo)):
        embryo['Cell'][i].printCell()

##############################################################################################################
# Reinicialize Predictors
##############################################################################################################

def reinitializePredictors():
    predictors = pd.DataFrame(columns = ['id', 'Predictor'])
    return predictors

##############################################################################################################
# Update KDTree with points from last image
##############################################################################################################

def updateKDTree(data, metric_distance):
    global kdt
    kdt = KDTree(data, metric = metric_distance)

##############################################################################################################
# Search for the match points using KDTree
##############################################################################################################

def getMatchingPoints(query_data, kelements, metric_distance):
    global kdt
    dist, elements = kdt.query(query_data, k = kelements, return_distance = True)
    result = pd.DataFrame({'distance':dist[:, 0], 'CellID':elements[:, 0]})
    return result

##############################################################################################################
# Deactivate unused cells in the current iteraction
##############################################################################################################

def deactivateUnusedCells(S1, activeCells):
    for i in range(0, len(S1)):
        if (S1[i] == False):
            embryo.active[activeCells.index[i]] = False
            # print('Deactivated!')
            
##############################################################################################################
# Define class to store information of the embryo
##############################################################################################################

class Cell:
    def __init__(self, cellID, begin, end):
        self.cellID = cellID
        self.begin = begin
        self.end = end
        self.parent = -1
        self.nseeds = 0
        self.seeds = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])
    
    def addSeed(self, time, x, y, z, _size):
        self.seeds.loc[self.nseeds] = [self.nseeds, time, x, y, z, _size]
        self.nseeds += 1
        self.end += 1
        
    def addSeedDf(self, nseed):
        self.seeds.loc[self.nseeds] = [self.nseeds, nseed.time, nseed.x, nseed.y, nseed.z, nseed._size]
        self.nseeds += 1
        self.end += 1
        
    def setEndPoint(self, end):
        self.end = end
    
    def setParentId(self, parent):
        self.parent = parent
    
    def getLastSeed(self):
        return self.seeds.loc[self.nseeds - 1]
    
    def getNumberOfSeeds(self):
        return self.nseeds
    
    def getCellID(self):
        return self.cellID
    
    def getSeedsDf(self):
        return self.seeds
    
    def printCellSummary(self):
        print(('CellID:', self.cellID,
                 'CellParentID:', self.parent,
                 'Begin:', self.begin,
                 'End:', self.end,
                 'NSeeds:', self.nseeds))
    
    def printCell(self):
        print(('CellID:', self.cellID,
                 'CellParentID:', self.parent,
                 'Begin:', self.begin,
                 'End:', self.end,
                 'NSeeds:', self.nseeds))
        
        print(self.seeds)

##############################################################################################################
# Define class for Polynomial Predictor
# This version uses Lagrange's polynomial
##############################################################################################################

class PolynomialPredictor():
    def __init__(self):
        self.averageDistance = 0.
        self.stdDevDistance = 0.
        self.averageAngle = 0.
        self.stdDevAngle = 0.
        self.error = 0.
        self.xyzpoints = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
    
    def predict(self):
        predictedPoint = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
        
        # Compute predicted point for each coordinate
        xPos = self.getPoint(self.xyzpoints['xpos'])
        yPos = self.getPoint(self.xyzpoints['ypos'])
        zPos = self.getPoint(self.xyzpoints['zpos'])
        
        # If predicted points are negative, set value to zero
        xPos = xPos if (xPos >= 0) else 0
        yPos = yPos if (yPos >= 0) else 0
        zPos = zPos if (zPos >= 0) else 0
        
        # Set the predicted points to the output vector
        predictedPoint.loc[0] = [xPos, yPos, zPos]
        
        # Return predicted coordinates
        return predictedPoint
    
    # Get the next point of the sequence by interpolating the input points and
    # extrapolating for the prediction of the next point
    def getPoint(self, points):
        # Number of points
        n = len(points)
        
        predPoint = n  # Point to be predicted (the next one in the sequence)
        polyNumer = 1. # Numerator of the polynomial equation
        polyDenom = 1. # Denominator of the polynomial equation
        function  = 0. # Polynomial equation result
        
        # Compute the movement (displacement) for each dimension (x, y and z)
        for i in range(0, n):
            polyNumer = 1.
            polyDenom = 1.
            
            for j in range(0, n):
                if (i != j):
                    polyNumer *= (predPoint - j)
                    polyDenom *= (i - j)
            
            # Compute polynomial equation
            function += ((polyNumer / polyDenom) * points.iloc[i])
        
        # Return predicted point (casted to int)
        return int(function)
    
    # Update points of the window and re-compute the statistics (error)
    def renewWindow(self, xyzpoints):
        # Update points of the window
        self.xyzpoints = xyzpoints
        
        # Update error with points of the current window
        self.updateError()
    
    # Update error of the interpolation
    def updateError(self):
        n = len(self.xyzpoints)
        distSum = 0.
        distSum2 = 0.
        
        distances = pd.DataFrame(columns= ['EuclDistance'])
        angles = pd.DataFrame(columns = ['Angle'])

        # Initialize variable
        initialVector = [0, 1, 0]
        
        for i in range(0, (n - 1)):
            
            p1 = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
            p1.loc[i] = [self.xyzpoints['xpos'].iloc[i],
                         self.xyzpoints['ypos'].iloc[i],
                         self.xyzpoints['zpos'].iloc[i]]
            
            p2 = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
            p2.loc[i] = [self.xyzpoints['xpos'].iloc[i + 1],
                         self.xyzpoints['ypos'].iloc[i + 1],
                         self.xyzpoints['zpos'].iloc[i + 1]]
            
            distances.loc[i] = [dist.euclidean(p1, p2)]
            
            distSum += distances['EuclDistance'][i]

            vector = self.getVectorCenteredInOrign(p1, p2)
            if (i > 0):
                # Add angles in radians
                angles.loc[i-1] = [self.getAngleBetween(initialVector, vector)]
            
            initialVector = vector

        # Compute average distance between the points
        average = 0.
        if (len(distances) > 0):
            average = distSum / len(distances)
        else:
            average = 0.
            
        for i in range(0, len(distances)):
            distSum2 += math.pow(distances['EuclDistance'][i] - average, 2)
        
        # Compute standard deviation of the points
        stdeviation = 0.
        if (len(distances) > 1):
            stdeviation = distSum2 / (len(distances) - 1)
        else:
            stdeviation = 0.
        
        # Compute the average and std deviation of the angle between points
        averageAngle = 0.
        stdDevAngle = 0.
        
        if (len(angles) > 0):
            averageAngle = angles.mean()[0]
            stdDevAngle = angles.std()[0]
        else:
            averageAngle = 0.
            stdDevAngle = 0.

        # Update statistics of the predictor
        self.averageDistance = average
        self.stdDevDistance = stdeviation
        self.averageAngle = averageAngle
        self.stdDevAngle = stdDevAngle
    
    # CHECK THE DIRECTION OF THE PREDICTED VECTOR
    # Computes the unit vector of the input vector
    def getUnitVector(self, inputvector):
        if (np.linalg.norm(inputvector) > 0):
            return inputvector / np.linalg.norm(inputvector)
        else:
            return ([0.] * len(inputvector))
        
    # Computes the angle (in radians) between the input vectors 'v1' and 'v2'
    def getAngleBetween(self, vector1, vector2):
        v1_unit = np.squeeze(np.asarray(self.getUnitVector(vector1)))
        v2_unit = np.squeeze(np.asarray(self.getUnitVector(vector2)))

        # Return the angle in radians
        return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))

    def getVectorCenteredInOrign(self, p1, p2):
        vector = [np.abs(p2['xpos'] - p1['xpos']),
                  np.abs(p2['ypos'] - p1['ypos']),
                  np.abs(p2['zpos'] - p1['zpos'])]
        
        return vector
    # =================================================================================

    # Add a new point to the window. This point will be used in the prediction
    # of the next point, along with the other points already in the window.
    def incrementWindow(self, newPx, newPy, newPz):
        self.xyzpoints.loc[len(self.xyzpoints)] = [newPx, newPy, newPz]

    # Check if error is below the threshold. Return 1 if the error is equal or
    # below the (average + deviation), and 0 otherwise
    def checkError(self, x, y, z):
        npoint = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
        npoint.loc[0] = [float(x), float(y), float(z)]
        
        n = len(self.xyzpoints) - 1
        
        lastpoint = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
        lastpoint.loc[0] = [float(self.xyzpoints['xpos'].iloc[n]),
                            float(self.xyzpoints['ypos'].iloc[n]),
                            float(self.xyzpoints['zpos'].iloc[n])]
        
        nangle = -1
        # There are at least two previous points (to get the direction)
        if (n > 0):
            lastpoint.loc[1] = [float(self.xyzpoints['xpos'].iloc[n-1]),
                                float(self.xyzpoints['ypos'].iloc[n-1]),
                                float(self.xyzpoints['zpos'].iloc[n-1])]
            
            previousVector = self.getVectorCenteredInOrign(lastpoint.iloc[1], lastpoint.iloc[0])
            nvector = self.getVectorCenteredInOrign(npoint.iloc[0], lastpoint.iloc[0])

            nangle = self.getAngleBetween(previousVector, nvector)

        # Compute distance between new point and the last one in the sequence
        distance = dist.euclidean(npoint.iloc[0], lastpoint.iloc[0])

        # Changed this error: check it
        if (distance <= (np.abs(self.averageDistance + self.stdDevDistance))):
            
            if (nangle > -1):
                # Compare the new angle with the average inside the window
                if (np.abs(nangle) <= (np.abs(self.averageAngle + self.stdDevAngle))):
                    return 1 # Error inside the accepted error bound
                else:
                    return 0 # Error is too much
            else:
                return 1 # Error inside the accepted error bound (without considering the angle)
        
        return 0 # Error is too much
            
    
    def getInitialPoint(self):
        initialPoint = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
        initialPoint.loc[0] = [self.xyzpoints['xpos'][0],
                               self.xyzpoints['ypos'][0],
                               self.xyzpoints['zpos'][0]]
                
        return initialPoint
    
    def getLastPoint(self):
        n = len(self.xyzpoints) - 1
                
        lastpoint = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
        lastpoint.loc[0] = [self.xyzpoints['xpos'][n],
                            self.xyzpoints['ypos'][n],
                            self.xyzpoints['zpos'][n]]
        
        return lastpoint
    
    def getNumberOfPoints(self):
        return (len(self.xyzpoints))


##############################################################################################################
# Read input metadata from (first) cells
##############################################################################################################

def getTwangData(filepath, filenames):
    global twangData
    # Read file names which contains the detected cells at each iteraction
    twangData = pd.read_csv(filepath + filenames, ';')
    twangData.rename(index = str, columns = {'size': '_size'}, inplace = True)
    return twangData

##############################################################################################################
# Add first cells into the embryo
# In this part, each seed is considered a new cell, since this is the first image processed
##############################################################################################################

def initializeEmbryo(filepath, filenames):
    global embryo
    global seedAtCell
    global twangData
    global buffer
    global numberOfNonPredictions
    global predictors
    global output

    seedAtCell = pd.DataFrame(columns = ['CellID'])
    twangData = getTwangData(filepath, filenames[0][0])

    # Create cells and add them to the embryo
    for i in range(0, len(twangData)):
        # Create cell
        ncell = Cell(i, 0, -1)
        # Add seed to the new cell
        ncell.addSeed(0, twangData.xpos[i], twangData.ypos[i], twangData.zpos[i], twangData._size[i])
        # A new cell of the embryo is set as active
        embryo.loc[i] = [ncell, True]
        # Store the cell which each new seed point was assigned to
        seedAtCell.loc[i] = [i]

    output.loc[0] = [0, len(embryo), sum(embryo.active == True)]
    buffer += 1
    numberOfNonPredictions += 1

##############################################################################################################
# Track cells while buffer is not full
##############################################################################################################

def track(i, filepath, filenames):
    global seedAtCell
    global twangData
    global embryo
    global numberOfNonPredictions
    global buffer
    global output
    
    # Read input data from this iteration
    twangData = pd.read_csv(filepath + filenames[0][i], ';')
    twangData.rename(index = str, columns = {'size': '_size'}, inplace = True)
    
    # Number of active cells at last iteration
    nCells = len(seedAtCell)
    
    # Get the new seeds data into the proper format
    newSeeds = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])
    
    # Get the new seeds of this point of time
    for ns in range(0, len(twangData)):
        newSeeds.loc[ns] = [-1, i, twangData.xpos[ns], twangData.ypos[ns], twangData.zpos[ns], twangData._size[ns]]
    
    # Get the matching cells for each new seed point
    matches = getMatchingPoints(query_data = twangData[['xpos', 'ypos', 'zpos']].copy(),
                                kelements = 1,
                                metric_distance = 'euclidean')
    
    childrenPerCell = [0] * len(embryo)
    
    if (len(matches) != len(newSeeds)):
        print('Something went wrong with the number of new seeds. An error is expected.')
    
    # Counter the number of matching seeds for each existing cell
    for ns in range(len(newSeeds)):
        childrenPerCell[seedAtCell['CellID'][matches['CellID'][ns]]] += 1

    updateSeedAtCell = pd.DataFrame(columns = ['CellID'], data = matches['CellID'])
    
    # Add seeds to the existent cells
    # or create new cells as the results of a cell division
    for ns in range(len(newSeeds)):
        
        if (childrenPerCell[seedAtCell['CellID'][matches['CellID'][ns]]] == 1):        
            # Add the only matching seed to the corresponding cell
            embryo.Cell[seedAtCell['CellID'][matches['CellID'][ns]]].addSeedDf(newSeeds.loc[ns])
            updateSeedAtCell['CellID'][ns] = seedAtCell['CellID'][matches['CellID'][ns]]
        else:
            if (childrenPerCell[seedAtCell['CellID'][matches['CellID'][ns]]] > 1):
                # More than one match
                ncell = Cell(len(embryo), (i), (i))
                ncell.addSeedDf(newSeeds.loc[ns])
                # Set matching cell as the new cell's parent
                ncell.setParentId(embryo.Cell[seedAtCell['CellID'][matches['CellID'][ns]]].getCellID())
                embryo.loc[len(embryo)] = [ncell, True]
                
                # This position is related to the last cell added to the embryo
                updateSeedAtCell['CellID'][ns] = len(embryo) - 1

    # Set unused cells as inactive
    for ec in range(len(childrenPerCell)):
        if (childrenPerCell[ec] != 1):
            embryo.active[ec] = False

    seedAtCell = updateSeedAtCell.copy()
    output.loc[i] = [i, len(embryo), sum(childrenPerCell)]
    
    buffer += 1
    numberOfNonPredictions += 1

    return (buffer, numberOfNonPredictions)

##############################################################################################################
# Get active cells to perform prediction
# Get active cells from embryo, given the seed-to-cell correspondence from last KDTree-related iteration
##############################################################################################################

def getActiveCells(seedAtCell):
    global embryo

    activeCells = pd.DataFrame(columns = ['index', 'Cell', 'active'])
    
    for i in range(len(seedAtCell)):
        # If cell is inactive, it has <1 seed, so it cannot be used for prediction
        if(embryo.loc[seedAtCell['CellID'][i]]['active'] == True):
            activeCells.loc[i] = [seedAtCell['CellID'][i],
                                  embryo.loc[seedAtCell['CellID'][i]]['Cell'],
                                  embryo.loc[seedAtCell['CellID'][i]]['active']]
    
    activeCells.reset_index(inplace = True, drop = True)
    return activeCells

##############################################################################################################
# Get last predicted points to use when returning to the tracking part
# These seeds are going to be used to construct the current KDTree structure
##############################################################################################################

def getLastPredictedSeeds(seedAtCell):
    new_seeds = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
    
    for i in range(len(seedAtCell)):
        # If this cell has only one seed, it wasn't used to predict
        if (embryo.Cell[seedAtCell['CellID'][i]].getNumberOfSeeds() > 1 and
            embryo.loc[seedAtCell['CellID'][i]]['active'] == True):
            last_seed = embryo.Cell[seedAtCell['CellID'][i]].getLastSeed()
        
            new_seeds.loc[i] = [last_seed['x'], last_seed['y'], last_seed['z']]
    
    return new_seeds

##############################################################################################################
# Initialize window of points and Predict
##############################################################################################################

def initializeAndPredict(buffer, numberOfPredictions, isWindowInitialized):
    global embryo
    global activeCells
    global predictors
    
    # Initialize window with the w last points
    # Use only cells with more than one seed
    newPredictors = pd.DataFrame(columns = ['id', 'Predictor'])
    
    # Track where to insert each new seed
    
    # Active cells in the embryo
    activeCells = getActiveCells(seedAtCell)
        
    # Number of active cells at last iteration
    nCells = len(activeCells)
    
    for i in range(0, nCells):
        # Check if this cell has more than one seed and more than the size of the window (w)
        if (activeCells.Cell[i].getNumberOfSeeds() > 1 and activeCells.Cell[i].getNumberOfSeeds() > w):
            # Get last w seed of the cell vector and add them to the predictor
            seeds = activeCells.Cell[i].getSeedsDf()
            begin = len(seeds) - w
            end = len(seeds)
            seedsW = seeds[begin:end].copy()
            seedsW.reset_index(inplace = True, drop = True)
            # Get points of seeds per coordinate
            xPoints = seedsW['x']
            yPoints = seedsW['y']
            zPoints = seedsW['z']
                        
            windowPoints = pd.concat([xPoints, yPoints, zPoints], axis = 1, sort = False)
            windowPoints.rename(index=str, columns={'x': 'xpos', 'y': 'ypos', 'z' : 'zpos'}, inplace=True)
            
            nPred = PolynomialPredictor()
            nPred.renewWindow(windowPoints)
            
            # Predict next point
            npoint = nPred.predict()
            
            # Add predicted point to the window
            nPred.incrementWindow(npoint['xpos'][0], npoint['ypos'][0], npoint['zpos'][0])
            
            # Add predictor of this cell to the list of predictors
            newPredictors.loc[len(newPredictors)] = [len(newPredictors), nPred]
            
            # Create new seed with the predicted point
            nSeed = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])
            nSeed.loc[0] = [-1, len(output), npoint['xpos'][0], npoint['ypos'][0], npoint['zpos'][0], 0]
            
            # Add the predicted point to the current cell
            embryo.Cell[activeCells['index'][i]].addSeedDf(nSeed.loc[0])
        else:
            # Add all seeds to the predictor (number of seeds is minor than w)
            if (activeCells.Cell[i].getNumberOfSeeds() > 1):
                nPred = PolynomialPredictor()
                seeds = activeCells.Cell[i].getSeedsDf()
                
                # Get the points of the seeds per coordinate
                xPoints = seeds['x']
                yPoints = seeds['y']
                zPoints = seeds['z']

                windowPoints = pd.concat([xPoints, yPoints, zPoints], axis = 1, sort = False)
                windowPoints.rename(index=str, columns={'x': 'xpos', 'y': 'ypos', 'z' : 'zpos'}, inplace=True)
                # Add points to the window
                nPred.renewWindow(windowPoints)
                
                # Predict next point
                npoint = nPred.predict()
                
                # Add predicted point to the window
                nPred.incrementWindow(npoint['xpos'][0], npoint['ypos'][0], npoint['zpos'][0])
                
                # Add predictor of this cell to the list of predictors
                newPredictors.loc[len(newPredictors)] = [len(newPredictors), nPred]

                # Create new seed with the predicted point
                nSeed = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])
                nSeed.loc[0] = [-1, len(output), npoint['xpos'][0], npoint['ypos'][0], npoint['zpos'][0], 0]

                # Add the predicted point to the current cell
                embryo.Cell[activeCells['index'][i]].addSeedDf(nSeed.loc[0])
                
            else:
                # In this case, the cell has only one seed, so it will be ignored (deactivated)
                embryo.active[activeCells['index'][i]] = False # Set cell as inactive
    
    predictors = reinitializePredictors()
    
    for p in range(0, len(newPredictors)):
        predictors.loc[p] = [p, newPredictors['Predictor'].loc[p]]
    
    isWindowInitialized = True # Set window as already initialized
    numberOfPredictions += 1 # Increment number of predictions
    
    return (buffer, numberOfPredictions, isWindowInitialized, predictors)

##############################################################################################################
# Predict, the window of points is already full
##############################################################################################################

def predict(i, buffer, numberOfPredictions, isWindowInitialized):
    global embryo
    global activeCells
    
    # Active cells in the embryo
    activeCells = getActiveCells(seedAtCell)
    
    # Number of active cells at last iteration
    nCells = len(activeCells)
    
    # Track cells that will continue to be active, and those which are inactive
    isValid = [False] * nCells
    
    sumError = 0.
    newPoints = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
    
    for nc in range(0, nCells):
        # At least two seeds to predict the next one
        if (activeCells.Cell[nc].getNumberOfSeeds() > 1):
            npoint = predictors.Predictor[nc].predict()
            finalp = predictors.Predictor[nc].getLastPoint()
            
            d = dist.euclidean([npoint['xpos'][0], npoint['ypos'][0], npoint['zpos'][0]],
                               [finalp['xpos'][0], finalp['ypos'][0], finalp['zpos'][0]])

            errOutput = predictors.Predictor[i].checkError(npoint['xpos'], npoint['ypos'], npoint['zpos'])
            sumError += errOutput
            
            newPoints.loc[len(newPoints)] = [npoint.xpos[0], npoint.ypos[0], npoint.zpos[0]]
            isValid[nc] = True
        else:
            # This cell is going to be set as inactive
            isValid[nc] = False
    
    relError = ((sumError * 100) / nCells) / 100
    relError = 1 - relError
    
    if (relError < maxError):
        # Add the predicted points to the cells
        for nc in range(0, nCells):
            # If this cell has a prediction
            if (isValid[nc] == True):
                predictors.Predictor[nc].incrementWindow(newPoints['xpos'][nc], newPoints['ypos'][nc], newPoints['zpos'][nc])
                
                # Create new seed with the predicted point
                nSeed = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])
                nSeed.loc[0] = [-1, len(output), newPoints['xpos'][nc], newPoints['ypos'][nc], newPoints['zpos'][nc], 0]
                
                # Add the predicted point to the current cell
                embryo.Cell[activeCells['index'][nc]].addSeedDf(nSeed.loc[0])
            else:
                # Set cell as inactive (it has only one seed, CMP cannot interpolate it)
                embryo.Cell[activeCells['index'][nc]].active = False # Set cell as inactive
                
        numberOfPredictions += 1
    else:
        # Error is beyond the acceptable amount. Renew window
        buffer = math.ceil(buffer * pw)
        isWindowInitialized = False
    
    return (buffer, numberOfPredictions, isWindowInitialized)

##############################################################################################################
# Track and estimate cells' trajectories
##############################################################################################################

def trackAndPredict(filepath, filenames):
    global seedAtCell
    global embryo
    global twangData
    global kdt
    global isWindowInitialized
    global predictors
    global numberOfNonPredictions
    global numberOfPredictions
    global buffer
    
    # The first position of the vector was already processed
    sequenceSize = len(filenames) # Number of iteractions

    # Use this flag to know if we are tracking (1) or predicting (2)
    flag_t_or_p = 1

    # For each point in time/in the sequence
    i = 1
    while i < sequenceSize:
        
        # If buffer is not full yet, perform tracking and increment buffer
        if (buffer < w):
            
            if (flag_t_or_p == 1):
                # Update KDTree with points of the last processed image
                updateKDTree(data = twangData[['xpos', 'ypos', 'zpos']].copy(), metric_distance = 'euclidean')
            else:
                # TODO: check if cells marked as inactive are going to be used after prediction
                # They should not!
                
                # Since we predicted points in the last iteration, the kdtree structure
                # will be constructed using the last predicted points of active cells
                td = getLastPredictedSeeds(seedAtCell)
                updateKDTree(data = td[['xpos', 'ypos', 'zpos']].copy(), metric_distance = 'euclidean')
                # Update flag, since we are going to track actual points in this iteration
                flag_t_or_p = 1

            
            buffer, numberOfNonPredictions = track(i, filepath, filenames)
        else:
            flag_t_or_p = 2
            
            # Check if window needs to be initialized
            if (isWindowInitialized == True):
                
                # Predict points using PolynomialPredictor
                buffer, numberOfPredictions, isWindowInitialized = predict(i, buffer, numberOfPredictions, isWindowInitialized)
                
                if (isWindowInitialized == False):
                    # Error is too much, predict cells for this time point
                    i -= 1
                    
            else:
                buffer, numberOfPredictions, isWindowInitialized, predictors = initializeAndPredict(buffer, numberOfPredictions, isWindowInitialized)            
                
                if (isWindowInitialized == False):
                    # Error is too much, predict cells for this time point
                    i -= 1
                
                flag_t_or_p = 2
                    
    #     print(i, len(embryo), sum(embryo.active == True))
        output.loc[i] = [i, len(embryo), sum(embryo.active == True)]
        i += 1

##############################################################################################################
# Remove orphan cells
##############################################################################################################

def removeOrphanCells(output, embryo, minSizeToRemove = 3):
    output_withoutOrphans = output.copy()

    for i in range(len(embryo)):
        if (embryo['Cell'].iloc[i].nseeds < minSizeToRemove):
            for c in range(embryo['Cell'].iloc[i].begin, len(output_withoutOrphans)):
                output_withoutOrphans['total_cells'].loc[c] = output_withoutOrphans['total_cells'].iloc[c] - 1

    return output_withoutOrphans

##############################################################################################################
# Print cells to test the range of distances and plot trajectories
##############################################################################################################

# Build a DataFrame to print cells' seed positions, for plotting
def getCellPositions(embryo):
    positions = pd.DataFrame(columns = ['CellID', 'SeedID', 'time', 'x', 'y', 'z', 'seedSize'])

    for i in range(len(embryo)):
        seeds = embryo.Cell[i].getSeedsDf()
        nseeds = embryo.Cell[i].getNumberOfSeeds()
            
        for s in range(nseeds):
            positions.loc[len(positions)] = [embryo.Cell[i].getCellID(), seeds['idSeed'][s], seeds['time'][s],
                                            seeds['x'][s], seeds['y'][s], seeds['z'][s], seeds['_size'][s]]

    return positions

##############################################################################################################
# Main
##############################################################################################################

def main(argv):
    global w
    global pw
    global maxError
    global output
    global embryo

    w               = int(argv[1])
    pw              = float(argv[2])
    maxError        = float(argv[3])
    filenames_path  = argv[4]
    files_path      = argv[5]
    output_path     = argv[6]
    
    buffer = 0
    numberOfNonPredictions = 0
    isWindowInitialized = False
    predictors = pd.DataFrame(columns = ['id', 'Predictor'])

    filenames = pd.read_csv(filenames_path, header = None)
    initializeEmbryo(files_path, filenames)
    trackAndPredict(files_path, filenames)
    output = composeOutputVector(len(filenames))
    
    # Generate output without orphan cells
    output_withoutOrphans = removeOrphanCells(output, embryo, minSizeToRemove = 3)
    
    # Save file with the number of cells per timestamp
    output.columns = ['imageId', 'totalCells', 'activeCells']
    output[['imageId', 'activeCells', 'totalCells']].to_csv(output_path, index = False)

    # Save output file without Orphan Cells
    output_withoutOrphans.columns = ['imageId', 'totalCells', 'activeCells']
    output_withoutOrphans[['imageId', 'activeCells', 'totalCells']].to_csv(output_path + '_withoutOrphans.csv', index = False)

    # Get and save file with cell positions
    positions = getCellPositions(embryo)
    positions.to_csv(output_path + '_embryo_cellPositions.csv', index=False)

if __name__ == "__main__":
    if (len(sys.argv) != 7):
        print("Wrong number of input parameters.")
        print('Usage: <w> <pw> <maxError> <input file with filenames> <path to input files> <output_path>')
    else:
        print('Estimating cells\' trajectories using KDTree-Predictor...')
        main(sys.argv)
        print('Done.')
