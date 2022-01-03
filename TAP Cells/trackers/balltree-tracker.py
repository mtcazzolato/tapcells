### 
# BallTree-Tracker: tracking cells from a developing embryo
# using the BallTree structure.
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
# BallTree-Tracker
##############################################################################################################

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as dist
from sklearn.neighbors import BallTree

##############################################################################################################
# Parameters and Data Structure
##############################################################################################################

# Set of cells from an embryo
embryo = pd.DataFrame(columns = ['Cell', 'active'])

# Output vector
output = pd.DataFrame(columns = ['id', 'total_cells', 'active_cells'])

##############################################################################################################
# Update BallTree with points from last image
##############################################################################################################

def updateBallTree(data, metric_distance):
    global blt
    blt = BallTree(data, metric = metric_distance)

##############################################################################################################
# Search for the match points using BallTree
##############################################################################################################

def getMatchingPoints(query_data, kelements, metric_distance):
    global blt
    dist, elements = blt.query(query_data, k = kelements, return_distance = True)
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
        print(('CellID:', self.cellID, 'CellParentID:', self.parent, 'Begin:', self.begin, 'End:', self.end, 'NSeeds:', self.nseeds))
    
    def printCell(self):
        print(('CellID:', self.cellID, 'CellParentID:', self.parent, 'Begin:', self.begin, 'End:', self.end, 'NSeeds:', self.nseeds))
        print(self.seeds)

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
    global seedAtCell
    global twangData
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

##############################################################################################################
# Track where to insert each new seed
##############################################################################################################

def track(filepath, filenames):
    # seedAtCell comes from the last iteration, and stores the cell id in order for each point in the BallTree
    global seedAtCell
    global twangData

    # The first position of the vector was already processed
    sequenceSize = len(filenames) # Number of iteractions

    for i in range(1, sequenceSize):
        # Update BallTree with points of the last processed image
        updateBallTree(data = twangData[['xpos', 'ypos', 'zpos']].copy(), metric_distance = 'euclidean')
        
        # Read input data from this iteration
        twangData = pd.read_csv(filepath + filenames[0][i], ';')
        twangData.rename(index = str, columns = {'size': '_size'}, inplace = True)
        
        # Number of active cells at last iteration
        nCells = len(seedAtCell)

        # Get the new seeds data into the proper format
        newSeeds = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])

        # Create the new seeds of this point of time
        for ns in range(0, len(twangData)):
            newSeeds.loc[ns] = [-1, i, twangData.xpos[ns], twangData.ypos[ns], twangData.zpos[ns], twangData._size[ns]]

        # Get the matching cells for each new seed point
        matches = getMatchingPoints(query_data = twangData[['xpos', 'ypos', 'zpos']].copy(),
                                    kelements = 1,
                                    metric_distance = 'euclidean')

        childrenPerCell = [0] * len(embryo)

        if (len(matches) != len(newSeeds)):
            print('Something is wrong with the number of new seeds. An error is expected.')

        # Count the number of matching seeds for each existing cell
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
    filenames_path = argv[1]
    files_path     = argv[2]
    output_path    = argv[3]

    filenames = pd.read_csv(filenames_path, header = None)
    initializeEmbryo(files_path, filenames)
    track(files_path, filenames)
    
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
    if (len(sys.argv) != 4):
        print('Wrong number of input parameters.')
        print('Usage: <input file with filenames> <path to input files> <output_path>')
    else:
        print('Tracking using BallTree-Tracker...')
        main(sys.argv)
        print('Done.')
