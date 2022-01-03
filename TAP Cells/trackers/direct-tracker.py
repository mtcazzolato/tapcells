### 
# Direct-Tracker: tracking cells from a developing embryo.
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
# Direct-Tracker
##############################################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as dist

##############################################################################################################
# Parameters and Data Structure
##############################################################################################################

# Distance threshold
# th = 18

# Set of cells from an embryo
embryo = pd.DataFrame(columns = ['Cell', 'active'])

# Output vector
output = pd.DataFrame(columns = ['id', 'total_cells', 'active_cells'])

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
# Deactivate unused cells in the current iteraction
##############################################################################################################

def deactivateUnusedCells(S1, activeCells):
    for i in range(0, len(S1)):
        if (S1[i] == False):
            embryo.active[activeCells.index[i]] = False
            print('Deactivated!')

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
# FIRST Read of input metadata from cells
##############################################################################################################

def getTwangData(filepath, filenames):
    # Read file names which contains the detected cells at each iteraction
    twangData = pd.read_csv(filepath + filenames, ';')
    twangData.rename(index = str, columns = {'size': '_size'}, inplace = True)
    return twangData

##############################################################################################################
# Add first cells into the embryo
# In this part, each seed is considered a new cell, since this is the first image processed
##############################################################################################################

def initializeEmbryo(filepath, filenames):
    data = getTwangData(filepath, filenames[0][0])

    # Create cells and add them to the embryo
    for i in range(0, len(data)):
        # Create cell
        ncell = Cell(i, 0, -1)
        # Add seed to the new cell
        ncell.addSeed(0, data.xpos[i], data.ypos[i], data.zpos[i], data._size[i])
        # A new cell of the embryo is set as active
        embryo.loc[i] = [ncell, True]

    output.loc[0] = [0, len(embryo), sum(embryo.active == True)]

##############################################################################################################
# Track where to insert each new seed
##############################################################################################################

def track(filepath, filenames):
    # The first position of the vector was already processed
    sequenceSize = len(filenames) # Number of iteractions

    # For each point in time/in the sequence
    for i in range(1, sequenceSize):
        # Read input data from this iteration
        twangData = getTwangData(filepath, filenames[0][i])
        twangData.rename(index = str, columns = {'size': '_size'}, inplace = True)

        # Active cells in the embryo
        activeCells = embryo[embryo.active == True]
        activeCells.reset_index(inplace = True)

        # Number of active cells at last iteration
        nCells = len(activeCells)

        # Get the new seeds data into the proper format
        newSeeds = pd.DataFrame(columns = ['idSeed', 'time', 'x', 'y', 'z', '_size'])

        # Create the new seeds of this point of time
        for ns in range(0, len(twangData)):
            newSeeds.loc[ns] = [-1, i, twangData.xpos[ns], twangData.ypos[ns], twangData.zpos[ns], twangData._size[ns]]

        S1 = [-1] * len(activeCells) # Each position refers to an existent (and active) cell
        S2 = [-1] * len(newSeeds)    # Each position refers to a new seed point

        # Store the number of children of a given cell
        childrenPerCell = [0] * len(activeCells)

        # For each new seed point at time i
        for ns in range(0, len(newSeeds)):
            cellIndex = -1
            found = False

            # Search for the first seed point within th and assin is as a match
            for ac in range(0, len(activeCells)):
                if (found == False):
                    d = computeDistanceCoordinates(newSeeds.loc[ns], activeCells.Cell[ac].getLastSeed(), dist.euclidean)
                    
                    if (d < th):
                        cellIndex = ac # Store the index of the cell that is a probable match
                        ac = len(activeCells) # Go to next seed (this one already has a match)
                        found = True

            # Store the id of the matching cell to this new seed (-1 is there isn't a match)
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
                # Increment counter with the number of children of the matching cell
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
                embryo.loc[activeCells['index'][c]].active = False

        output.loc[i] = [i, len(embryo), sum(embryo.active == True)]

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
    global embryo
    
    th = int(argv[1])
    filenames_path = argv[2]
    files_path = argv[3]
    output_path = argv[4]

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
    if (len(sys.argv) != 5):
        print('Wrong number of input parameters.')
        print('Usage: <th> <input file with filenames> <path to input files> <output_path>')
    else:
        print('Tracking using Direct-Tracker...')
        main(sys.argv)
        print('Done.')
