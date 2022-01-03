### 
# TAP Cells: Tracking and prediction of cell positions.
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

import sys
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from tkinter import *
from pandastable import Table, TableModel

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import help
import about

def set_Tk_var():
    global varLabelOutputResultsTable
    varLabelOutputResultsTable = tk.StringVar()
    varLabelOutputResultsTable.set('Label')

    global varLabelOutputGTTable
    varLabelOutputGTTable = tk.StringVar()
    varLabelOutputGTTable.set('Cells per timestamp (from Ground Truth):')

    global varRedInformation
    varRedInformation = tk.StringVar()
    varRedInformation.set('Please, inform the task (Track or Predict).')

    global varResultSeqSize
    varResultSeqSize = tk.StringVar()
    varResultSeqSize.set('--')

    global varResultTDR
    varResultTDR = tk.StringVar()
    varResultTDR.set('--')

    global varResultTCRE
    varResultTCRE = tk.StringVar()
    varResultTCRE.set('--')

    global varResultTotalCells
    varResultTotalCells = tk.StringVar()
    varResultTotalCells.set('--')

    global varTotalCellsGT
    varTotalCellsGT = tk.StringVar()
    varTotalCellsGT.set('--')

    global varResultExecTime
    varResultExecTime = tk.StringVar()
    varResultExecTime.set('--')

    global varEntryTrackerTh
    varEntryTrackerTh = tk.StringVar()

    global varEntryPredMaxError
    varEntryPredMaxError = tk.StringVar()

    global varEntryPredTh
    varEntryPredTh = tk.StringVar()

    global varEntryPredW
    varEntryPredW = tk.StringVar()

    global varEntryPredPw
    varEntryPredPw = tk.StringVar()

    global varEntryPredDegree
    varEntryPredDegree = tk.StringVar()

    global varLabelCommand
    varLabelCommand = tk.StringVar()
    varLabelCommand.set('')

    global VarGTFile
    VarGTFile = tk.StringVar()
    VarGTFile.set('Select ground truth file')

    global VarInputFile
    VarInputFile = tk.StringVar()
    VarInputFile.set('Select input file')

    global VarOutputFile
    VarOutputFile = tk.StringVar()
    VarOutputFile.set('Select output file')

    global VarInputPath
    VarInputPath = tk.StringVar()
    VarInputPath.set('Select input path')

    global predictorsList
    predictorsList = tk.StringVar()
    predictorsList.set(["CM-Predictor", "KDT-Predictor", "BLT-Predictor", "Int-Predictor"])

    global trackersList
    trackersList = tk.StringVar()
    trackersList.set(["Direct-Tracker", "Clever-Tracker", "KDT-Tracker", "BLT-Tracker"])
    
    global selectedButton
    selectedButton = tk.IntVar()

    global CanvasPlot

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top
    
    # Set default values for combo boxes
    w.TComboboxPredictors.current(0)
    w.TComboboxTrackers.current(0)

    global taskID # 0=track, 1=predict, -1=undefined
    taskID = -1

    varLabelOutputGTTable.set('')
    varLabelOutputResultsTable.set('')

def btnSelectInput():
    fname = tk.filedialog.askopenfile(mode="r",
                                      initialdir = "./",
                                      title = "Select file",
                                      filetypes = (("text files","*.txt"),("csv files","*.csv"),("all files","*.*")))
    VarInputFile.set(fname.name)
    sys.stdout.flush()

def btnSelectOutput():
    fnameoutput = tk.filedialog.asksaveasfilename(initialdir = "./",
                                                  title = "Save output as",
                                                  filetypes = (("csv file", ".csv"),("text file","*.txt"),("all files","*.*")))
    VarOutputFile.set(fnameoutput)
    sys.stdout.flush()

def btnSelectPath():
    dirname = tk.filedialog.askdirectory(initialdir = "./", title = "Input directory")
    VarInputPath.set(dirname + '/')
    sys.stdout.flush()
    
def trackersSelect(event):
    index = event.widget.curselection()[0]
    value = event.widget.get(index)

def predictorsSelect(event):
    index = event.widget.curselection()[0]
    value = event.widget.get(index)

def getAlgorithmFileName(algorithm):
    algDict = {'Direct-Tracker' : 'direct-tracker.py',
               'Clever-Tracker' : 'clever-tracker.py',
               'KDT-Tracker' : 'kdtree-tracker.py',
               'BLT-Tracker' : 'balltree-tracker.py',
               'CM-Predictor' : 'cm-predictor.py',
               'BLT-Predictor (Lagrange)'   : 'balltree-predictor.py',
               'KDT-Predictor (Lagrange)'   : 'kdtree-predictor.py',
               'KDT-Predictor (Polynomial)' : 'kdtree-predictor-regression.py',
               'KDT-Predictor (Weighted Polynomial)' : 'kdtree-predictor-regression-weighted.py',
               'KDT-Predictor (Lagrange) AvgError'   : 'kdtree-predictor_avgError.py',
               'KDT-Predictor (Polynomial) AvgError' : 'kdtree-predictor-regression_avgError.py',
               'KDT-Predictor (Weighted Polynomial) AvgError' : 'kdtree-predictor-regression-weighted_avgError.py',
               'KDT-Predictor (Lagrange) AngleError'   : 'kdtree-predictor_directionError.py',
               'KDT-Predictor (Polynomial) AngleError' : 'kdtree-predictor-regression_directionError.py',
               'KDT-Predictor (Weighted Polynomial) AngleError' : 'kdtree-predictor-regression-weighted_directionError.py',
               'Int-Predictor (Lagrange)'   : 'interleaved-predictor.py',
               'Int-Predictor (Polynomial)' : 'interleaved-predictor-regression.py',
               'Int-Predictor (Weighted Polynomial)' : 'interleaved-predictor-regression-weighted.py'}

    return algDict[algorithm]

def BtnRunAlgorithmClick():
    global taskID, CanvasPlot

    try: # To allow reploting charts
        CanvasPlot.get_tk_widget().destroy()
    except:
        pass
    
    if(taskID == -1):
        print('Error: Select a Tracker or Predictor first.')
    else:
        algorithm = ''
        
        if(taskID == 0):
            algorithm = w.TComboboxTrackers.get()
            # print('Prediction with', algorithm)
        elif(taskID == 1):
            algorithm = w.TComboboxPredictors.get()
            # print('Tracking with', algorithm)
    
        inputFile = VarInputFile.get()
        inputPath = VarInputPath.get()
        outputFile = VarOutputFile.get()
        gtFile = VarGTFile.get()

        commandString = ''

        if (algorithm in ['Direct-Tracker', 'Clever-Tracker']): # Trackers that require th parameter
            commandString = ('python3 trackers/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryTrackerTh.get()) +
                             ' \'' + inputFile + '\' \'' + inputPath +
                             '\' \'' + outputFile + '\'')
            
        elif (algorithm in ['KDT-Tracker', 'BLT-Tracker']): # Parameter-free trackers
            commandString = ('python3 trackers/' + getAlgorithmFileName(algorithm) +
                             ' \'' + inputFile + '\' \'' + inputPath
                             + '\' \'' + outputFile + '\'')

        elif (algorithm in ['CM-Predictor', 'BLT-Predictor (Lagrange)', 'KDT-Predictor (Lagrange)']): # Predictors with Lagrange's
            r = 1. # Relaxation parameter

            commandString = ('python3 predictors/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryPredTh.get()) + ' ' + str(varEntryPredW.get()) +
                             ' ' + str(varEntryPredPw.get()) + ' ' + str(varEntryPredMaxError.get()) +
                             ' ' + str(r) + ' \'' + inputFile + '\' \'' + inputPath + '\' \'' +
                             outputFile + '\'')
            
        elif (algorithm in ['KDT-Predictor (Polynomial)']): # Predictor with polynomial regression
            r = 1. # Relaxation parameter

            commandString = ('python3 predictors/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryPredTh.get()) + ' ' + str(varEntryPredW.get()) +
                             ' ' + str(varEntryPredPw.get()) + ' ' + str(varEntryPredMaxError.get()) +
                             ' ' + str(r) + ' \'' + inputFile + '\' \'' + inputPath + '\' \'' +
                             outputFile + '\' ' + str(varEntryPredDegree.get()))
            
        elif (algorithm in ['KDT-Predictor (Weighted Polynomial)']): # Predictor with weighted polynomial regression
            r = 1. # Relaxation parameter

            commandString = ('python3 predictors/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryPredTh.get()) + ' ' + str(varEntryPredW.get()) +
                             ' ' + str(varEntryPredPw.get()) + ' ' + str(varEntryPredMaxError.get()) +
                             ' ' + str(r) + ' \'' + inputFile + '\' \'' + inputPath + '\' \'' +
                             outputFile + '\' ' + str(varEntryPredDegree.get()) + ' True')
            
        elif (algorithm in ['Int-Predictor (Lagrange)']):
            commandString = ('python3 predictors/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryPredW.get()) +
                             ' \'' + inputFile + '\' \'' + inputPath + '\' \'' +
                             outputFile + '\'')
            
        elif (algorithm in ['Int-Predictor (Polynomial)']):
            commandString = ('python3 predictors/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryPredW.get()) +
                             ' \'' + inputFile + '\' \'' + inputPath + '\' \'' +
                             outputFile + '\' ' + str(varEntryPredDegree.get()))
            
        elif (algorithm in ['Int-Predictor (Weighted Polynomial)']):
            commandString = ('python3 predictors/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryPredW.get()) +
                             ' \'' + inputFile + '\' \'' + inputPath + '\' \'' +
                             outputFile + '\' ' + str(varEntryPredDegree.get()) + ' True')
        
        elif (algorithm in ['KDT-Predictor (Lagrange) AvgError', 'KDT-Predictor (Lagrange) AngleError']):
            commandString = ('python3 predictors/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryPredW.get()) + ' ' + str(varEntryPredPw.get()) +
                             ' ' + str(varEntryPredMaxError.get()) +
                             ' \'' + inputFile + '\' \'' + inputPath + '\' \'' + outputFile + '\'')

        elif (algorithm in ['KDT-Predictor (Polynomial) AvgError', 'KDT-Predictor (Polynomial) AngleError']):
            commandString = ('python3 predictors/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryPredW.get()) + ' ' + str(varEntryPredPw.get()) +
                             ' ' + str(varEntryPredMaxError.get()) +
                             ' \'' + inputFile + '\' \'' + inputPath + '\' \'' +
                             outputFile + '\' ' + str(varEntryPredDegree.get()))

        elif (algorithm in ['KDT-Predictor (Weighted Polynomial) AvgError', 'KDT-Predictor (Weighted Polynomial) AngleError']):
            commandString = ('python3 predictors/' + getAlgorithmFileName(algorithm) +
                             ' ' + str(varEntryPredW.get()) + ' ' + str(varEntryPredPw.get()) +
                             ' ' + str(varEntryPredMaxError.get()) +
                             ' \'' + inputFile + '\' \'' + inputPath + '\' \'' +
                             outputFile + '\' ' + str(varEntryPredDegree.get()) + ' True')
        
        exec_time = -1
        
        try:
            start_time = time.time()
            os.system(commandString)
            exec_time = (time.time() - start_time)
            print('Cell tracking/prediction finished.')
        except:
            print('Error executing the algorithm.')
            pass

        try: # Show cell positions outputted by the algorithm
            dfPositions = pd.read_csv(outputFile + '_embryo_cellPositions.csv')
            tablePositions = Table(w.TFrameCellPositions,
                                    dataframe = dfPositions,
                                    showtoolbar = True,
                                    showstatusbar = True)
            tablePositions.show()
            tablePositions.redraw()
        except:
            print('Error acessing output file with cell positions.')
            pass

        try: # Show output number of cells
            dfOutputs = pd.read_csv(outputFile)
            tableOutputs = Table(w.TFrameOutputs,
                                    dataframe = dfOutputs,
                                    showtoolbar = True,
                                    showstatusbar = True)
            tableOutputs.show()
            tableOutputs.redraw()
        except:
            print('Error acessing output file with number of cells.')
            pass
        
        try: # Show ground truth data
            dfGT = pd.read_csv(gtFile)
            tableGT = Table(w.TFrameGroundTruth,
                                    dataframe = dfGT,
                                    showtoolbar = True,
                                    showstatusbar = True)
            tableGT.show()
            tableGT.redraw()
        except:
            print('Error acessing file with ground truth information.')
            pass
        
        try: # Show generated trajectories
            # Set the plot parameters
            sns.set()
            figure, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(dfPositions['x'].values, dfPositions['y'].values, c = dfPositions['CellID'].values)
            ax.set_xlabel('x position')
            ax.set_ylabel('y position')
            # plt.tight_layout()
            figure.tight_layout()
            plt.show()
            
            CanvasPlot = FigureCanvasTkAgg(figure, master=w.TFrameTrajectoryPlot)
            CanvasPlot.draw()
            CanvasPlot.get_tk_widget().pack(side=tk.TOP, fill=tk.X)

        except:
            print('Error showing plot with generated trajectories.')
            pass
    
        try: # Show statistics
            varLabelCommand.set('Executed command: ' + commandString)
            varResultExecTime.set('{:.2f} s'.format(exec_time))
            varResultSeqSize.set(len(dfOutputs))
            varResultTotalCells.set(dfOutputs['totalCells'].iloc[-1])
            varTotalCellsGT.set(dfGT['totalCells'].iloc[-1])
            # Compute Tracking Detection Rate (TDR)
            tdr = computeTDR(dfOutputs['totalCells'].values, dfGT['totalCells'].values, len(dfOutputs))
            varResultTDR.set('{:.4f}'.format(tdr[-1]))
            # Compute Total Cell Relative Error (TCRE)
            tcre = abs(1 - (dfOutputs['totalCells'].iloc[-1] / dfGT['totalCells'].iloc[-1]))
            varResultTCRE.set('{:.4f}'.format(tcre))
        except:
            pass
    
    varLabelOutputGTTable.set('Cells per timestamp (from Ground Truth):')
    varLabelOutputResultsTable.set('Cells per timestamp (from tracking/prediction):')

    root.mainloop()
    sys.stdout.flush()

def computeTDR(ground_truth, tracker, nimages):
    diff_ground_truth = getDifference(ground_truth)
    diff_tracker = getDifference(tracker)
    gtXtracker = (diff_ground_truth == diff_tracker)
    
    tdr_gtXtracker = np.array([None] * nimages)
    
    for i in range (0, nimages):
        tdr_gtXtracker[i] = sum(gtXtracker[0:(i + 1)]) / float(i + 1)
    return tdr_gtXtracker

def getDifference(tracker):
    diff_tracker = tracker[1:] - tracker[0:-1]
    return diff_tracker
    
def RadioButtonPredSelect():
    global taskID
    taskID = 1

    # Enable Prediction entries/cbox and disable Tracking ones
    w.TComboboxPredictors.configure(state='readonly')
    w.TComboboxTrackers.configure(state='disable')
    
    ComboboxPredictorsSelected(None)
    
    sys.stdout.flush()

def RadioButtonTrackingSelect():
    global taskID
    taskID = 0

    # Enable Prediction entries/cbox and disable Tracking ones
    w.TComboboxPredictors.configure(state='disable')
    w.TComboboxTrackers.configure(state='readonly')

    # Treat value selection of the corresponding combobox
    ComboboxTrackersSelected(None)
    sys.stdout.flush()

def btnSelectGTFile():
    gtname = tk.filedialog.askopenfile(mode="r",
                                        initialdir = "./",
                                        title = "Select file",
                                        filetypes = (("text files","*.txt"),("csv files","*.csv"),("all files","*.*"))
                                    )
    VarGTFile.set(gtname.name)
    sys.stdout.flush()

def menuAboutClick():
    about.create_ToplevelAbout(root)
    sys.stdout.flush()

def menuExitClick():
    sys.stdout.flush()
    sys.exit()

def menuHelpClick():
    help.create_ToplevelHelp(root)
    sys.stdout.flush()

def disableEntries():
    w.EntryTrackerTh.configure(state='disable')
    w.EntryPredMaxError.configure(state='disable')
    w.EntryPredTh.configure(state='disable')
    w.EntryPredW.configure(state='disable')
    w.EntryPredPw.configure(state='disable')
    w.EntryPredDegree.configure(state='disable')

def ComboboxTrackersSelected(event):
    disableEntries()
    algorithm = w.TComboboxTrackers.get()
    msg = ''
    
    # Treat entries and messages
    if (algorithm in ['Direct-Tracker', 'Clever-Tracker']):
        msg = (algorithm + ' - inform parameter: <th>')
        w.EntryTrackerTh.configure(state='normal')
    else:
        msg = (algorithm + ' is parameter-free.')

    varRedInformation.set(msg)
    sys.stdout.flush()

def ComboboxPredictorsSelected(event):
    disableEntries()
    algorithm = w.TComboboxPredictors.get()
    msg = ''
    
    # Treat entries and messages
    if (algorithm in ['CM-Predictor', 'BLT-Predictor (Lagrange)', 'KDT-Predictor (Lagrange)']):
        msg = (algorithm + ' - inform parameters: <th> <w> <pw> <maxError>')
        w.EntryPredMaxError.configure(state='normal')
        w.EntryPredTh.configure(state='normal')
        w.EntryPredW.configure(state='normal')
        w.EntryPredPw.configure(state='normal')
        
    elif (algorithm in ['KDT-Predictor (Polynomial)', 'KDT-Predictor (Weighted Polynomial)']): # Predictor with polynomial regression
        msg = (algorithm + ' - inform parameters: <th> <w> <pw> <maxError> <degree>')
        w.EntryPredMaxError.configure(state='normal')
        w.EntryPredTh.configure(state='normal')
        w.EntryPredW.configure(state='normal')
        w.EntryPredPw.configure(state='normal')
        w.EntryPredDegree.configure(state='normal')
    
    elif (algorithm in ['KDT-Predictor (Lagrange) AvgError', 'KDT-Predictor (Lagrange) AngleError']):
        msg = (algorithm + ' - inform parameters: <w> <pw> <maxError>')
        w.EntryPredMaxError.configure(state='normal')
        w.EntryPredW.configure(state='normal')
        w.EntryPredPw.configure(state='normal')
    
    elif (algorithm in ['KDT-Predictor (Polynomial) AvgError',
                        'KDT-Predictor (Weighted Polynomial) AvgError',
                        'KDT-Predictor (Polynomial) AngleError',
                        'KDT-Predictor (Weighted Polynomial) AngleError']): # Predictor with polynomial regression
        msg = (algorithm + ' - inform parameters: <w> <pw> <maxError> <degree>')
        w.EntryPredMaxError.configure(state='normal')
        w.EntryPredW.configure(state='normal')
        w.EntryPredPw.configure(state='normal')
        w.EntryPredDegree.configure(state='normal')
    
    elif (algorithm in ['Int-Predictor (Lagrange)']):
        msg = (algorithm + ' - inform parameter: <w>')
        w.EntryPredW.configure(state='normal')
    
    elif (algorithm in ['Int-Predictor (Polynomial)', 'Int-Predictor (Weighted Polynomial)']):
        msg = (algorithm + ' - inform parameters: <w> <degree>')
        w.EntryPredW.configure(state='normal')
        w.EntryPredDegree.configure(state='normal')

    else:
        msg = (algorithm + ' is parameter-free.')

    varRedInformation.set(msg)

    sys.stdout.flush()

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import tapcells
    tapcells.vp_start_gui()





