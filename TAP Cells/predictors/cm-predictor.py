### 
# CM-Predictor: tracking and predicting cells from a developing embryo.
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
# CM-Predictor
##############################################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as dist
import math

##############################################################################################################
# Parameters and Data Structure
##############################################################################################################

# Distance threshold
# th = 18

# Set of cells from the embryo
embryo = pd.DataFrame(columns = ['Cell', 'active'])

# Set of predictors for the cells
predictors = pd.DataFrame(columns = ['id', 'Predictor'])

# Output vector
output = pd.DataFrame(columns = ['id', 'total_cells', 'active_cells'])

# CM-Predictor's parameters:
# w                # Window size
# pw               # Percentage of the window to be discarded
# maxError         # Maximum error allowed for prediction
# r                # Relaxation parameter

# CM-Predictor's control variables:
buffer = 0
numberOfPredictions = 0
numberOfNonPredictions = 0
isWindowInitialized = False

##############################################################################################################
# Reinitialize predictors
##############################################################################################################

def reinitializePredictors():
    predictors = pd.DataFrame(columns = ['id', 'Predictor'])
    return predictors

##############################################################################################################
# Compute distance between two seed points
##############################################################################################################

# s1 and s2 are two dataframes, each one containing a feature vector, and both of same size
def computeDistance(s1, s2, distance):
    if (len(s1.values) != len(s2.values)):
        print('ERROR: Feature vectors must be of same size.')
        return -1
    
    dist = distance(s1.values, s2.values)
    return dist

# s1 and s2 are two dataframes, each one containing a feature vector, and both of same size
def computeDistanceCoordinates(s1, s2, distance):
    if (len(s1.values) != len(s2.values)):
        print('ERROR: Feature vectors must be of same size.')
        return -1

    dist = distance([s1.x, s1.y, s1.z], [s2.x, s2.y, s2.z])
    return dist

##############################################################################################################
# Deactivate unused cells in the current iteraction
##############################################################################################################

def deactivateUnusedCells(S1, activeCells):
    for i in range(0, len(S1)):
        if (S1[i] == False):
            embryo.active[activeCells.index[i]] = False

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
    
    def getSeeds(self):
        return self.seeds

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
        self.averageDistance = 0
        self.stdDevDistance = 0
        self.error = 0
        self.xyzpoints = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
#        self.xpoints = pd.DataFrame(columns='xpoint')
#        self.ypoints = pd.DataFrame(columns='xpoint')
#        self.zpoints = pd.DataFrame(columns='xpoint')
    
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
        
#        # Return predicted point
#        return function
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
        
        for i in range(0, (n - 1)):
            
            p1 = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
            p1.loc[i] = [self.xyzpoints['xpos'][i],
                         self.xyzpoints['ypos'][i],
                         self.xyzpoints['zpos'][i]]
            
            p2 = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
            p2.loc[i] = [self.xyzpoints['xpos'][i + 1],
                         self.xyzpoints['ypos'][i + 1],
                         self.xyzpoints['zpos'][i + 1]]
            
            distances.loc[i] = [dist.euclidean(p1, p2)]
            
            distSum += distances['EuclDistance'][i]
        
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
        
        # Update statistics of the predictor
        self.averageDistance = average
        self.stdDevDistance = stdeviation
    
    # Add a new point to the window. This point will be used in the prediction
    # of the next point, along with the other points already in the window.
    def incrementWindow(self, newPx, newPy, newPz):
        self.xyzpoints.loc[len(self.xyzpoints)] = [newPx, newPy, newPz]
    
    # Check if error is below the threshold. Return 1 if the error is equal or
    # below the (average + deviation), and 0 otherwise
    def checkError(self, x, y, z):
        npoint = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
        npoint.loc[0] = [x, y, z]
        
        n = len(self.xyzpoints) - 1
        
        lastpoint = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
        lastpoint.loc[0] = [self.xyzpoints['xpos'][n],
                            self.xyzpoints['ypos'][n],
                            self.xyzpoints['zpos'][n]]
        
        # Compute distance between new point and the last one in the sequence
        distance = dist.euclidean(npoint, lastpoint)
        
#        print(...)
        
        if (distance <= (math.pow((self.averageDistance + self.stdDevDistance), 2))):
            return 1 # Error inside the accepted error bound
        
        return 0
            
    
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
    # Read file names which contains the detected cells at each iteraction
    # filenames = pd.read_csv('~/experiments/extension-tracking/exp11_filenames.txt', header = None)
    # filepath = '~/experiments/extension-tracking/'

    twangData = pd.read_csv(filepath + filenames, ';')
    twangData.rename(index = str, columns = {'size': '_size'}, inplace = True)
    return twangData

##############################################################################################################
# Add first cells into the embryo
##############################################################################################################

def initializeEmbryo(filepath, filenames, buffer, numberOfNonPredictions):
    twangData = getTwangData(filepath, filenames[0][0])

    # Create cells and add them to the embryo
    for i in range(0, len(twangData)):
        # Create cell
        ncell = Cell(i, 0, -1)
        # Add seed to the new cell
        ncell.addSeed(0, twangData.xpos[i], twangData.ypos[i], twangData.zpos[i], twangData._size[i])
        # A new cell of the embryo is set as active
        embryo.loc[i] = [ncell, True]

    output.loc[0] = [0, len(embryo), sum(embryo.active == True)]

    buffer += 1
    numberOfNonPredictions += 1

    return (buffer, numberOfNonPredictions)

##############################################################################################################
# Track cells while buffer is not full
##############################################################################################################

def track(i, filepath, filenames, embryo, buffer, numberOfNonPredictions):
    # Read input data from this iteration
    twangData = pd.read_csv(filepath + filenames[0][i], ';')
    twangData.rename(index = str, columns = {'size': '_size'}, inplace = True)
    
    # Active cells in the embryo
    activeCells = embryo[embryo.active == True]
    activeCells.reset_index(inplace = True)
    
    # Number of active cells at last iteration
    nCells = len(activeCells)
    
    # Get the new seeds data into the proper format
    newSeeds = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])
    
    # Get the new seeds of this point of time
    for ns in range(0, len(twangData)):
        newSeeds.loc[ns] = [-1, i, twangData.xpos[ns], twangData.ypos[ns], twangData.zpos[ns], twangData._size[ns]]
    
    S1 = [-1] * len(activeCells) # Each position refers to an existent (and active) cell
    S2 = [-1] * len(newSeeds)    # Each position refers to a new seed point
    
    # Store the number of childen of a given cell
    childrenPerCell = [0] * len(activeCells)
    
    # For each new seed point at time i
    for ns in range(0, len(newSeeds)):
        cellIndex = -1
        found = False
        distance = math.inf # Assign the equivalent to +infinity (maximum float value)
        
        # Search for the first seed point within th an assign it as a match
        for ac in range(0, len(activeCells)):
            d = computeDistanceCoordinates(newSeeds.loc[ns], activeCells.Cell[ac].getLastSeed(),
                                           dist.euclidean)
            
            if (distance > d and d < th):
                distance = d
                # Store the index of the cell that is a probable match
                cellIndex = ac
        
        # Store the id of the matching cell to this new seed (-1 if there isn't a match)
        S2[ns] = cellIndex
        
    # Now add the seeds with no cell matching to the embryo
    # and count the number of new seeds for each existing cell
    for ns in range(0, len(newSeeds)):
        if (S2[ns] == -1): # Seed without a matching cell, add it as a new cell
            # Cell index it the last one of the embryo
            cellIndex = len(embryo)
            # Create cell (with the beginning and ending on the current time)
            ncell = Cell(cellIndex, (i), (i))
            # Add seed to the new cell
            ncell.addSeedDf(newSeeds.loc[ns])
            ncell.setParentId(0) # It has as its parent the first cell of the embryo
            # The new cell is set as active
            embryo.loc[len(embryo)] = [ncell, True]
        else:
            cellIndexMatching = S2[ns]
            # Increment counter with the number of children of the matchgetNumberOfSeedsing cell
            childrenPerCell[cellIndexMatching] += 1

    # Add seeds to the existent cells
    # or create new cells as the results of a cell division
    for ns in range(0, len(newSeeds)):
        cellIndexMatching = S2[ns]
  
        if (cellIndexMatching != -1): # Add seed to an existent cell
            if (childrenPerCell[cellIndexMatching] == 1):
                embryo.Cell[activeCells['index'][cellIndexMatching]].addSeedDf(newSeeds.loc[ns])
            else: # Create new cells as the results of a cell division
                if (childrenPerCell[cellIndexMatching] > 1):
                    ncell = Cell(len(embryo), (i), (i))
                    ncell.addSeedDf(newSeeds.loc[ns])
                    # Set matching cell as the new cell's parent
                    ncell.setParentId(embryo.Cell[activeCells['index'][cellIndexMatching]].getCellID())
                    embryo.loc[len(embryo)] = [ncell, True]

    # Set cells as inactive
    for c in range(0, len(activeCells)):
        if (childrenPerCell[c] != 1):
            #embryo.loc[activeCells['index'][c]].active = False
            embryo.active[activeCells['index'][c]] = False
    
    buffer += 1
    numberOfNonPredictions += 1
    
    return (buffer, numberOfNonPredictions)

##############################################################################################################
# Initialize window of points and Predict
##############################################################################################################

def predict(i, embryo, buffer, numberOfPredictions, isWindowInitialized, predictors):
    # Active cells in the embryo
    activeCells = embryo[embryo.active == True]
    activeCells.reset_index(inplace = True)
    
    # Number of active cells at last iteration
    nCells = len(activeCells)
    
    # Track cells that will continue to be active, and those wich are inactive
    isValid = [False] * nCells
    
    sumError = 0.
    newPoints = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
    
    for c in range(0, nCells):
        
        # At least two seeds to predict the next one
        if (activeCells.Cell[c].getNumberOfSeeds() > 1):
            npoint = predictors.Predictor[c].predict()
            finalp = predictors.Predictor[c].getLastPoint()
            
            d = dist.euclidean([npoint['xpos'][0], npoint['ypos'][0], npoint['zpos'][0]],
                               [finalp['xpos'][0], finalp['ypos'][0], finalp['zpos'][0]])

            if (d <= (th * r)):
                # Error inside the accepted error bound
                sumError += 1
            else:
                sumError += 0
            
            newPoints.loc[len(newPoints)] = [npoint.xpos[0], npoint.ypos[0], npoint.zpos[0]]
            isValid[c] = True
        else:
            # This cells is going to be set as inactive
            isValid[c] = False
    
    relError = 0.0
    if (nCells != 0):
        relError = ((sumError * 100) / nCells) / 100
    
    relError = 1 - relError

    if (relError < maxError):
        # Add the predicted points to the cells

        for c in range(0, nCells):
            # If this cell has a prediction
            if (isValid[c] == True):
                predictors.Predictor[c].incrementWindow(newPoints['xpos'][c], newPoints['ypos'][c], newPoints['zpos'][c])
                
                # Create new seed with the predicted point
                nSeed = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])
                nSeed.loc[0] = [-1, i, newPoints['xpos'][c], newPoints['ypos'][c], newPoints['zpos'][c], 0]
                
                # Add the predicted point to the current cell
                embryo.Cell[activeCells['index'][c]].addSeedDf(nSeed.loc[0])
            else:
                # Set cell as inactive (it has only one seed, CMP cannot interpolate it)
                embryo.Cell[activeCells['index'][c]].active = False # Set cell as inactive
                
        numberOfPredictions += 1
    else:
        # Error is beyond the acceptable amount. Renew window
        buffer = math.ceil(buffer * pw)
        isWindowInitialized = False
    
    return (predictors, buffer, numberOfPredictions, isWindowInitialized)

##############################################################################################################
# Predict, the window of points is already full
##############################################################################################################

def initializeAndPredict(i, embryo, buffer, numberOfPredictions, isWindowInitialized):
    # Initialize window with the w last points
    # Use only cells with more than one seed
    newPredictors = pd.DataFrame(columns = ['id', 'Predictor'])
    
    # Track where to insert each new seed
    
    # Active cells in the embryo
    activeCells = embryo[embryo.active == True]
    activeCells.reset_index(inplace = True)
    
    # Number of active cells at last iteration
    nCells = len(activeCells)
    
    for c in range(0, nCells):
        # Check if this cell has more than one seed and more than the size of the window (w)
        if (activeCells.Cell[c].getNumberOfSeeds() > 1 and activeCells.Cell[c].getNumberOfSeeds() > w):
            # Get last w seed of the cell vector and add them to the predictor
            seeds = activeCells.Cell[c].getSeeds()
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
            nSeed.loc[0] = [-1, i, npoint['xpos'][0], npoint['ypos'][0], npoint['zpos'][0], 0]
            
            # Add the predicted point to the current cell
            embryo.Cell[activeCells['index'][c]].addSeedDf(nSeed.loc[0])
        else:
            # Add all seeds to the predictor (number of seeds is minor than w)
            if (activeCells.Cell[c].getNumberOfSeeds() > 1):
                nPred = PolynomialPredictor()
                seeds = activeCells.Cell[c].getSeeds()
                
                # Get the points of the seeds per coordinate
                xPoints = seeds['x']
                yPoints = seeds['y']
                zPoints = seeds['z']

#                windowPoints = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
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
                nSeed.loc[0] = [-1, i, npoint['xpos'][0], npoint['ypos'][0], npoint['zpos'][0], 0]

                # Add the predicted point to the current cell
                embryo.Cell[activeCells['index'][c]].addSeedDf(nSeed.loc[0])
                
            else:
                # In this case, the cell has only one seed, so it will be ignored (deactivated)
                embryo.active[activeCells['index'][c]] = False # Set cell as inactive
    
    predictors = reinitializePredictors()
    
    for p in range(0, len(newPredictors)):
        predictors.loc[p] = [p, newPredictors['Predictor'].loc[p]]
    
    isWindowInitialized = True # Set window as already initialized
    numberOfPredictions += 1 # Increment number of predictions
    
    return (predictors, buffer, numberOfPredictions, isWindowInitialized, predictors)


##############################################################################################################
# Track and estimate cells' trajectories
##############################################################################################################

def trackAndPredict(files_path, filenames, buffer, numberOfPredictions, numberOfNonPredictions, predictors):
    # The first position of the vector was already processed
    sequenceSize = len(filenames) # Number of iteractions

    isWindowInitialized = False

    # For each point in time/in the sequence
    # for i in range(1, sequenceSize):
    i = 1
    while i < sequenceSize:
        # Read input data from this iteration
        twangData = getTwangData(files_path, filenames[0][i])
        twangData.rename(index = str, columns = {'size': '_size'}, inplace = True)

        # If buffer is not full yet, perform tracking and increment buffer
        if (buffer < w):
            buffer, numberOfNonPredictions = track(i, files_path, filenames,
                                                   embryo, buffer, numberOfNonPredictions)
        else:
            # Check if window needs to be initialized
            if (isWindowInitialized == True):
                # Predict points using PolynomialPredictor
                predictors, buffer, numberOfPredictions, isWindowInitialized = predict(i, embryo, buffer, numberOfPredictions, isWindowInitialized, predictors)

                if (isWindowInitialized == False):
                    # Error is too much, predict cells for this time point
                    i -= 1
            else:
                predictors, buffer, numberOfPredictions, isWindowInitialized, predictors = initializeAndPredict(i, embryo, buffer, numberOfPredictions, isWindowInitialized)            

                if (isWindowInitialized == False):
                    # Error is too much, predict cells for this time point
                    i -= 1
            
        #    print(i, len(embryo), sum(embryo.active == True))
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
    global th
    global w
    global pw
    global maxError
    global r
    global embryo
    th              = int(argv[1])
    w               = int(argv[2])
    pw              = float(argv[3])
    maxError        = float(argv[4])
    r               = float(argv[5])
    filenames_path  = argv[6]
    files_path      = argv[7]
    output_path     = argv[8]
    buffer = 0
    numberOfNonPredictions = 0

    filenames = pd.read_csv(filenames_path, header = None)
    buffer, numberOfNonPredictions = initializeEmbryo(files_path, filenames, buffer, numberOfNonPredictions)
    trackAndPredict(files_path, filenames, buffer, numberOfPredictions, numberOfNonPredictions, predictors)
    
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
    if (len(sys.argv) != 9):
        print("Wrong number of input parameters.")
        print('Usage: <th> <w> <pw> <maxError> <r> <input file with filenames> <path to input files> <output_path>')
    else:
        #print('Estimating cells\' trajectories using CM-Predictor...')
        main(sys.argv)
        #print('Done.')
