### 
# Interleaved-Predictor: interleaving tracking and predicting cells from a developing embryo
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
# Int-Predictor
##############################################################################################################

import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

##############################################################################################################
# Parameters and Data Structure
##############################################################################################################

# Set of cells from an embryo
embryo = pd.DataFrame(columns = ['Cell', 'active'])

# KDTree structures used in the match
# This tree matches the cells of first window (only at the first iteration)
# and holds the last cells of the first window at each iteration
global kdt
# This tree matches the cells of the second window
global kdt2

##############################################################################################################
# Print embryo information
##############################################################################################################

def printEmbryo():
    for pc in range(len(embryo)):
        embryo['Cell'].iloc[pc].printCell()

##############################################################################################################
# Update KDTree with points from last image
##############################################################################################################

def updateKDTree(data, metric_distance):
    global kdt
    kdt = KDTree(data, metric = metric_distance)

def updateKDTree2(data, metric_distance):
    global kdt2
    kdt2 = KDTree(data, metric = metric_distance)

##############################################################################################################
# Search the matching points using KDTree
##############################################################################################################

# For the first window:
def getMatchingPoints(query_data, kelements, metric_distance):
    global kdt
    dist, elements = kdt.query(query_data, k = kelements, return_distance = True)
    result = pd.DataFrame({'distance':dist[:, 0], 'CellID':elements[:, 0]})
    return result

# For the second window:
def getMatchingPoints2(query_data, kelements, metric_distance):
    global kdt2
    dist, elements = kdt2.query(query_data, k = kelements, return_distance = True)
    result = pd.DataFrame({'distance':dist[:, 0], 'CellID':elements[:, 0]})
    return result

##############################################################################################################
# Deactivate unused cells in the current iteration
##############################################################################################################

def deactivateUnusedCells(S1, activeCells):
    for i in range(0, len(S1)):
        if (S1[i] == False):
            embryo.active[activeCells.index[i]] = False
            # print('Deactivated!')

##############################################################################################################
# Get active cells to perform prediction
##############################################################################################################

## Get active cells from embryo, given the seed-to-cell correspondence from last KDTree-related iteration
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
# Reset seed at cell and data to index information
##############################################################################################################

def resetSeedAtCell(activeCells):
    seedAtCell = pd.DataFrame(columns = ['CellID'])
    dataToIndex = pd.DataFrame(columns = ['xpos', 'ypos', 'zpos'])
    
    for ac in range(len(activeCells)):
        seedAtCell.loc[ac] = [activeCells.Cell[ac].getCellID()]
        dataToIndex.loc[ac] = [activeCells.Cell[ac].getLastSeed()['x'],
                               activeCells.Cell[ac].getLastSeed()['y'],
                               activeCells.Cell[ac].getLastSeed()['z']]
    
    return seedAtCell, dataToIndex

##############################################################################################################
# Define Regression Function and Predict Next Point
##############################################################################################################

def predictNextPoint(previousTimestamps, previousPoints, newPosition, order, weighted):
    
    poly = PolynomialFeatures(degree = order)
    previousTimestamps_pr = poly.fit_transform(np.array(previousTimestamps).reshape(-1, 1))
    
    lr = LinearRegression()
    
    if (weighted == True):
        lr.fit(previousTimestamps_pr, previousPoints, sample_weight=previousTimestamps)
    else:
        lr.fit(previousTimestamps_pr, previousPoints)
    
    predictedPoint = lr.predict(poly.fit_transform(newPosition))
    
    # Return the predicted point
    return int(predictedPoint) # cast to int because lr.predict returns an array of one value for the input parameter

##############################################################################################################
# Define class to store the embryo information
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

    def setCellID(self, cellID):
        self.cellID = cellID

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

    def getInitialTime(self):
        return self.begin

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
# Read input metadata from (first) cells
##############################################################################################################

def getTwangData(filepath, filenames):
    # Read file names which contains the detected cells at each iteraction
    twangData = pd.read_csv(filepath + filenames, ';')
    twangData.rename(index = str, columns = {'size': '_size'}, inplace = True)
    return twangData

##############################################################################################################
# Problem: How do we match cells considering an interval?
# Step 1: Initialize first buffer
##############################################################################################################

def initializeFirstBuffer(filepath, filenames):
    global seedAtCell
    global twangData
    global buffer1
    global numberOfNonPredictions
    global embryo
    global w
    global kdt
    global childrenPerCell

    # Stores the reference of each seed point at each cell on the KDTree
    seedAtCell = pd.DataFrame(columns = ['CellID'])

    # Read data from first file
    twangData = getTwangData(filepath, filenames[0][0])

    # Create cells and add them to the embryo
    for i in range(0, len(twangData)):
        # Create cell
        ncell = Cell(i, 0, -1)
        # Add seed to the new cell
        ncell.addSeed(0,
                      twangData['xpos'].iloc[i],
                      twangData['ypos'].iloc[i],
                      twangData['zpos'].iloc[i],
                      twangData['_size'].iloc[i])
        # A new cell of the embryo is set as active
        embryo.loc[i] = [ncell, True]
        # Store the cell which each new seed point was assigned to
        seedAtCell.loc[i] = [i]

    # Increment first buffer, indicating that the first image/timestamnp was processed
    buffer1 += 1
    numberOfNonPredictions += 1

    # Add points to the embryo and buffer 1 (initialize first window)
    for it in range(1, w):
        
        # Update KDTree with points from this iteration
        updateKDTree(data = twangData[['xpos', 'ypos', 'zpos']].copy(), metric_distance = 'euclidean')
        
        # Read input data from this iteration
        twangData = getTwangData(filepath, filenames[0][it])
        
        # Number of active cells at last iteration
        nCells = len(seedAtCell)
        
        # Get the new seeds data into the proper format
        newSeeds = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])
        
        # Create the new seeds of this point of time
        for ns in range(len(twangData)):
            newSeeds.loc[ns] = [-1,
                                it,
                                twangData['xpos'].iloc[ns],
                                twangData['ypos'].iloc[ns],
                                twangData['zpos'].iloc[ns],
                                twangData['_size'].iloc[ns]]
        
        # Get the matching cells for each new seed point
        matches = getMatchingPoints(query_data = twangData[['xpos', 'ypos', 'zpos']].copy(),
                                   kelements = 1,
                                   metric_distance = 'euclidean')
        
        # Store, for each cell, the number of seeds resulting from it
        childrenPerCell = [0] * len(embryo)
        
        if (len(matches) != len(newSeeds)):
            print('Something is wrong with the number of new seeds. An error is expected.')
        
        # Note: seedAtCell is the reference of the seed data in the KDTree and the corresponding cell IDs
        # Count the number of matching seeds for each existing cell
        for ns in range(len(newSeeds)):
            # Get the id of the matching seed element of the KDTree
            matchingSeedID = matches['CellID'].iloc[ns]
            # Get the corresponding cell ID of the matching seed point of matchingSeedID
            correspondingCellID = seedAtCell['CellID'].iloc[matchingSeedID]
            # Increment the corresponding cell matching count
            childrenPerCell[correspondingCellID] += 1
        
        updateSeedAtCell = pd.DataFrame(columns = ['CellID'], data = matches['CellID'])
        
        # Add features to the existent cells
        # or create new cells as the results of a cell division
        for ns in range(len(newSeeds)):
            
            if (childrenPerCell[seedAtCell['CellID'].iloc[matches['CellID'][ns]]] == 1):
                # Add the only matching seed to the corresponding cell
                embryo.Cell[seedAtCell['CellID'].iloc[matches['CellID'][ns]]].addSeedDf(newSeeds.iloc[ns])
                
                updateSeedAtCell['CellID'].loc[ns] = seedAtCell['CellID'].iloc[matches['CellID'].iloc[ns]]
                
            elif (childrenPerCell[seedAtCell['CellID'].iloc[matches['CellID'].iloc[ns]]] > 1):
                # More than one match
                ncell = Cell(len(embryo), it, (it - 1))
                ncell.addSeedDf(newSeeds.loc[ns])
                # Set matching cell as the new cell's parent
                ncell.setParentId(embryo.Cell[seedAtCell['CellID'].iloc[matches['CellID'].iloc[ns]]].getCellID())
                embryo.loc[len(embryo)] = [ncell, True]

                # This position is related to the last cell added to the embryo
                updateSeedAtCell['CellID'].loc[ns] = len(embryo) - 1
        
        # Deactivate cells whith zero or more than one seed match(es)
        for dc in range(len(childrenPerCell)):
            if (childrenPerCell[dc] != 1):
                embryo.active[dc] = False
        
        seedAtCell = updateSeedAtCell.copy()
        
        buffer1 += 1
        numberOfNonPredictions += 1

        if (buffer1 == w):
            # Update KDTree with points of the last processed image of buffer1
            updateKDTree(data = twangData[['xpos', 'ypos', 'zpos']].copy(), metric_distance = 'euclidean')

##############################################################################################################
# Define matching and interpolation of intermediary points
# Match window points (in the extremities) and interpolate intermediary points
##############################################################################################################

def interpolateAndMatch(it, initialMatch):
    global seedAtCell
    global seedAtCell2
    global kdt
    global kdt2
    global pointsBf2
    global embryo
    global w
    global pointsBf2
    
    # Point in time to start interpolation (assuming the whole window2 was filled)
    initialTime = it - w
    
    childrenPerCell = [0] * len(embryo)
    
    # Count the number of matching seeds for each existing cell
    for ns in range(len(initialMatch)):
        childrenPerCell[seedAtCell['CellID'][initialMatch['CellID'][ns]]] += 1
    
    # For each point in the second window
    for i in range(len(pointsBf2)):
        if (pointsBf2['Cell'].iloc[i].getInitialTime() == initialTime):
            # Create interpolation array
            # The current cell will be the one that matches with the current position of pointsBf2
            currCell = embryo['Cell'][seedAtCell['CellID'][initialMatch['CellID'][i]]].getSeedsDf().copy()
            
            # We get only the first window of points (and not all points from the cell)
            if (len(currCell) > w):
                currCell = currCell[-w:]
                currCell.reset_index(inplace = True, drop = True)
            
            # Get points from the second window
            currCellBf2 = pointsBf2['Cell'][i].getSeedsDf().copy()
            
            # Get points from the second window
            interpDf = pd.concat([currCell, currCellBf2])

            # Check if the cell has only one match
            if (childrenPerCell[seedAtCell['CellID'][initialMatch['CellID'][i]]] == 1):
               # Add interpolated points to the embryo, in the corresponding cells
                for p in np.arange((it-(w*2)), (it-w)):
                                        
                    # Use weighted regression function to predict next point
                    px = predictNextPoint(interpDf['time'], # previous timestamps
                                                         interpDf['x'], # previous positions
                                                         p.reshape(1, -1), # next timestamp
                                                         order, # polynomial dregree
                                                         isWeighted) # True weights the points by position
                    py = predictNextPoint(interpDf['time'], # previous timestamps
                                                         interpDf['y'], # previous positions
                                                         p.reshape(1, -1), # next timestamp
                                                         order, # polynomial dregree
                                                         isWeighted) # True weights the points by position
                    pz = predictNextPoint(interpDf['time'], # previous timestamps
                                                         interpDf['z'], # previous positions
                                                         p.reshape(1, -1), # next timestamp
                                                         order, # polynomial dregree
                                                         isWeighted) # True weights the points by position
                    
                    embryo['Cell'][seedAtCell['CellID'][initialMatch['CellID'][i]]].addSeed(p, px, py, pz, 0)
                    
                # Add points from buffer 2 to the embryo
                for cbf2 in range(len(currCellBf2)):
                    embryo['Cell'][seedAtCell['CellID'][initialMatch['CellID'][i]]].addSeed(currCellBf2['time'].iloc[cbf2],
                                                                                               currCellBf2['x'].iloc[cbf2],
                                                                                               currCellBf2['y'].iloc[cbf2],
                                                                                               currCellBf2['z'].iloc[cbf2],
                                                                                               currCellBf2['_size'].iloc[cbf2])
            
                embryo.active[seedAtCell['CellID'][initialMatch['CellID'][i]]] = pointsBf2.active[i]

            # Check if the cell has more than one match
            elif (childrenPerCell[seedAtCell['CellID'][initialMatch['CellID'][i]]] > 1):
#                 print('Create a new cell.')
                # More than one match, create a new cell
                ncell = Cell(len(embryo), initialTime-w, initialTime-w-1)
                
                # Add interpolated points to the new cell
                for p in np.arange((it-(w*2)), (it-w)):
                    
                    # Use weighted regression function to predict next point
                    px = predictNextPoint(interpDf['time'], # previous timestamps
                                                         interpDf['x'], # previous positions
                                                         p.reshape(1, -1), # next timestamp
                                                         order, # polynomial dregree
                                                         isWeighted) # True weights the points by position
                    py = predictNextPoint(interpDf['time'], # previous timestamps
                                                         interpDf['y'], # previous positions
                                                         p.reshape(1, -1), # next timestamp
                                                         order, # polynomial dregree
                                                         isWeighted) # True weights the points by position
                    pz = predictNextPoint(interpDf['time'], # previous timestamps
                                                         interpDf['z'], # previous positions
                                                         p.reshape(1, -1), # next timestamp
                                                         order, # polynomial dregree
                                                         isWeighted) # True weights the points by position
                    ncell.addSeed(p, px, py, pz, 0)
                
                # Add points from buffer 2 to the new cell
                for cbf2 in range(len(currCellBf2)):
                    ncell.addSeed(currCellBf2['time'].iloc[cbf2],
                                  currCellBf2['x'].iloc[cbf2],
                                  currCellBf2['y'].iloc[cbf2],
                                  currCellBf2['z'].iloc[cbf2],
                                  currCellBf2['_size'].iloc[cbf2])
                
                # Set matching cell as the new cell's parent
                ncell.setParentId(embryo.Cell[seedAtCell['CellID'][initialMatch['CellID'][i]]].getCellID())
                embryo.loc[len(embryo)] = [ncell, pointsBf2['active'][i]]
                
            # Check if the cell has no match (this should not happen, since .getInitialTime == initialTime)
            else:
                print('An error has occurred. This cell should not exist.')
            
        else:
#             print('Add as new seed cell to the embryo')
            # Get points from the second window
            pointsBf2['Cell'].iloc[i].setCellID(len(embryo))
            embryo.loc[len(embryo)] = [pointsBf2['Cell'].iloc[i], pointsBf2['active'].iloc[i]]
            
#             if (pointsBf2['active'].iloc[i] == False):
#                 print('deactivated: CellID[', embryo.Cell[len(embryo) - 1].getCellID(), ']')

    # Deactivate (set as inactive) the unused cells, or the ones with more than one match (splitted ones)
    for ns in range(len(childrenPerCell)):
        if (childrenPerCell[ns] != 1):
            embryo['active'][ns] = False

##############################################################################################################
# Step 2. Read second window, match points to w1, and interpolate intermediary points
# Match cells from buffers, interpolate intermediary points and add them to the embryo
##############################################################################################################

def trackAndPredict(filepath, filenames):
    # seedAtCell comes from the last iteration, and stores the sorted cell id,
    # for each point in the KDTree
    global seedAtCell
    global twangData
    global twangData2
    global seedAtCell
    global seedAtCell2
    global activeCells
    global kdt
    global kdt2
    global initialMatch
    global buffer1
    global buffer2
    global numberOfPredictions
    global numberOfNonPredictions
    global w
    global childrenPerCell
    global pointsBf2
    global embryo
    
    pointsBf2 = pd.DataFrame(columns = ['Cell', 'active']) # Dataframe to store points from Buffer 2
    seedAtCell2 = pd.DataFrame(columns = ['CellID'])
    finalize = False
    sequenceSize = len(filenames) # Number of iterations
    it = w * 2 # 'it' starts with the index of the beginning of the second window

    while (it < sequenceSize):
        if (finalize == False):
            if (buffer2 == 0):
                # Read input data from this iteration
                twangData2 = getTwangData(filepath, filenames[0][it])
                
                # Get match of the initial points with kdtree1 to use after interpolation
                initialMatch = getMatchingPoints(query_data = twangData2[['xpos', 'ypos', 'zpos']].copy(),
                                                kelements = 1,
                                                metric_distance = 'euclidean')

                # Create cells and add them to the temporary embryo
                for cs in range(0, len(twangData2)):
                    # Create cell
                    ncell = Cell(cs, it, (it-1))
                    # Add seed to the new cell
                    ncell.addSeed(it, twangData2.xpos[cs], twangData2.ypos[cs], twangData2.zpos[cs], twangData2._size[cs])
                    # Create a temporary embryo
                    pointsBf2.loc[cs] = [ncell, True]
                    # Store the cell which each new seed point was assigned to
                    seedAtCell2.loc[cs] = [cs]
                
            elif (buffer2 < w): # Add points to the embryo and buffer2
                # Update KDTree with points of the last processed image
                updateKDTree2(data = twangData2[['xpos', 'ypos', 'zpos']].copy(),
                             metric_distance = 'euclidean')

                # Read input data from this iteration
                twangData2 = getTwangData(filepath, filenames[0][it])

                # Number of active cells at the last iteration
                nCells = len(seedAtCell2)

                # Get the new seeds data into the proper format
                newSeeds = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])

                # Get the matching cells for each seed point of buffer2
                for ns in range(0, len(twangData2)):
                    newSeeds.loc[ns] = [-1,
                                        it,
                                        twangData2.xpos[ns],
                                        twangData2.ypos[ns],
                                        twangData2.zpos[ns],
                                        twangData2._size[ns]]

                # Get the matching cells for each new seed point
                matches = getMatchingPoints2(query_data = twangData2[['xpos', 'ypos', 'zpos']].copy(),
                                            kelements = 1,
                                            metric_distance = 'euclidean')

                childrenPerCell = [0] * len(pointsBf2)

                if (len(matches) != len(newSeeds)):
                    print('Something is wrong with the number of new seeds. An error is expected.')

                # Note: seedAtCell2 is the reference of the seed data in the KDTree2 and the corresponding cell IDs
                # Count the number of matching seeds for each existing cell
                for ns in range(len(newSeeds)):
                    childrenPerCell[seedAtCell2['CellID'][matches['CellID'][ns]]] += 1

                updateSeedAtCell2 = pd.DataFrame(columns = ['CellID'], data = matches['CellID'])

                # Add seeds to the existent cells
                # or create new cells as the results of a cell division
                for ns in range(len(newSeeds)):
                    if (childrenPerCell[seedAtCell2['CellID'][matches['CellID'][ns]]] > 1):
                        # More than one match
                        ncell = Cell(len(pointsBf2), it, (it-1))
                        ncell.addSeedDf(newSeeds.loc[ns])
                        
                        # Set matching cell as the new cell's parent
                        ncell.setParentId(pointsBf2.Cell[seedAtCell2['CellID'][matches['CellID'][ns]]].getCellID())
                        pointsBf2.loc[len(pointsBf2)] = [ncell, True]

                        # This position is related to the last cell added to the embryo
                        updateSeedAtCell2['CellID'][ns] = len(pointsBf2) - 1

                    elif (childrenPerCell[seedAtCell2['CellID'][matches['CellID'][ns]]] == 1):
                        # Add the only matching seed to the corresponding cell
                        pointsBf2.Cell[seedAtCell2['CellID'][matches['CellID'][ns]]].addSeedDf(newSeeds.loc[ns])
                        updateSeedAtCell2['CellID'][ns] = seedAtCell2['CellID'][matches['CellID'][ns]]
                    
                seedAtCell2 = updateSeedAtCell2.copy()
                
            # Deactivate cells whith zero or more than one seed match(es)
            for dc in range(len(childrenPerCell)):
                if (childrenPerCell[dc] != 1):
                    pointsBf2.active[dc] = False
            
            numberOfNonPredictions += 1
            it += 1
            buffer2 += 1

            if ((buffer1 == buffer2)):
                # Interpolate
                # Add points of buffer2 to the embryo
                # Initialize buffer1 with points of buffer2
    #             print('buffers are equal')

                interpolateAndMatch(it, initialMatch)
                numberOfPredictions += w
                # Update KDTree with points of the last processed image
                
                # Update data to insert in the first KDTree
                activeCells = embryo[embryo.active == True]
                activeCells.reset_index(inplace = True, drop = True)

                seedAtCell, dataToIndex = resetSeedAtCell(activeCells)
                
                # Insert data from last iteration to the first KDTree
                updateKDTree(data = dataToIndex[['xpos', 'ypos', 'zpos']].copy(), metric_distance = 'euclidean')
                
                # Reset buffer cound and temporary embryo
                buffer2 = 0
                pointsBf2 = pd.DataFrame(columns = ['Cell', 'active'])
                
                if ( (it + w ) <= sequenceSize ):
                    it += w
                else:
                    finalize = True
        else: # finalize = True
            # Track remaining points
            # Read input data from this iteration
            twangData = getTwangData(filepath, filenames[0][it])
            
            # Number of active cells at last iteration
            nCells = len(seedAtCell)
            

            # Get the new seeds data into the proper format
            newSeeds = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])

            # Create the new seeds of this point of time
            for ns in range(len(twangData)):
                newSeeds.loc[ns] = [-1,
                                    it,
                                    twangData['xpos'].iloc[ns],
                                    twangData['ypos'].iloc[ns],
                                    twangData['zpos'].iloc[ns],
                                    twangData['_size'].iloc[ns]]

            # Get the matching cells for each new seed point
            matches = getMatchingPoints(query_data = twangData[['xpos', 'ypos', 'zpos']].copy(),
                                       kelements = 1,
                                       metric_distance = 'euclidean')

            # Store, for each cell, the number of seeds resulting from it
            childrenPerCell = [0] * len(embryo)

            if (len(matches) != len(newSeeds)):
                print('Something is wrong with the number of new seeds. An error is expected.')

            # Note: seedAtCell is the reference of the seed data in the KDTree and the corresponding cell IDs
            # Count the number of matching seeds for each existing cell
            for ns in range(len(newSeeds)):
                # Get the id of the matching seed element of the KDTree
                matchingSeedID = matches['CellID'].iloc[ns]
                # Get the corresponding cell ID of the matching seed point of matchingSeedID
                correspondingCellID = seedAtCell['CellID'].iloc[matchingSeedID]
                # Increment the corresponding cell matching count
                childrenPerCell[correspondingCellID] += 1

            updateSeedAtCell = pd.DataFrame(columns = ['CellID'], data = matches['CellID'])

            # Add features to the existent cells
            # or create new cells as the results of a cell division
            for ns in range(len(newSeeds)):

                if (childrenPerCell[seedAtCell['CellID'].iloc[matches['CellID'][ns]]] == 1):
                    # Add the only matching seed to the corresponding cell
                    embryo.Cell[seedAtCell['CellID'].iloc[matches['CellID'][ns]]].addSeedDf(newSeeds.iloc[ns])

                    updateSeedAtCell['CellID'].loc[ns] = seedAtCell['CellID'].iloc[matches['CellID'].iloc[ns]]

                elif (childrenPerCell[seedAtCell['CellID'].iloc[matches['CellID'].iloc[ns]]] > 1):
                    # More than one match
                    ncell = Cell(len(embryo), it, (it - 1))
                    ncell.addSeedDf(newSeeds.loc[ns])
                    # Set matching cell as the new cell's parent
                    ncell.setParentId(embryo.Cell[seedAtCell['CellID'].iloc[matches['CellID'].iloc[ns]]].getCellID())
                    embryo.loc[len(embryo)] = [ncell, True]

                    # This position is related to the last cell added to the embryo
                    updateSeedAtCell['CellID'].loc[ns] = len(embryo) - 1

            # Deactivate cells whith zero or more than one seed match(es)
            for dc in range(len(childrenPerCell)):
                if (childrenPerCell[dc] != 1):
                    embryo.active[dc] = False

            seedAtCell = updateSeedAtCell.copy()

            it += 1
            numberOfNonPredictions += 1
            
            if (it != sequenceSize):
                # Update KDTree with points of the last processed image of buffer1
                updateKDTree(data = twangData[['xpos', 'ypos', 'zpos']].copy(), metric_distance = 'euclidean')

    print('Done')


##############################################################################################################
# Compose output vector
##############################################################################################################

def composeOutputVector(sequenceSize):
    global embryo
    global output
    
    # Output vector
    output = pd.DataFrame(columns = ['id', 'total_cells', 'active_cells'])

    for t in range(sequenceSize):
        id = t
        total_cells = 0
        active_cells = 0
        
        for i in range(len(embryo)):
            if (t >= embryo['Cell'][i].begin):
                total_cells += 1
                if (t <= embryo['Cell'][i].end):
                    active_cells += 1
        
        output.loc[t] = [id, total_cells, active_cells]

    return output

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
    global numberOfPredictions
    global numberOfNonPredictions
    global buffer1
    global buffer2
    global kdt
    global kdt2
    global embryo
    global output
    global order
    global isWeighted

    w                   = int(argv[1]) # Window size
    filenames_path      = argv[2]
    files_path          = argv[3]
    output_path         = argv[4]
    order               = int(argv[5])
    # Cast boolean parameter
    isWeighted = (True if argv[6] == "True" else False)
    
    numberOfPredictions = 0
    numberOfNonPredictions = 0

    # Control of the windows of points
    buffer1 = 0
    buffer2 = 0

    filenames = pd.read_csv(filenames_path, header = None)

    initializeFirstBuffer(files_path, filenames)
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
        print('Usage: <w> <input file with filenames> <path to input files> <output_path> <order> <isWeighted>')
    else:
        print('Estimating cells\' trajectories using Interleaved-Predictor...')
        main(sys.argv)
        print('Done.')
