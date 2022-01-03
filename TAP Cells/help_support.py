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

def set_Tk_var():
    global varLabelParameters
    varLabelParameters = tk.StringVar()
    
    global varLabelFiles
    
    varLabelFiles = tk.StringVar()
    crMsg = ("- Input File: File with a list of files containing the cell"
             "positions previously extracted from the images.\n"
             "- Input Path: Path containing files listed in the input file.\n"
             "- Output File: File prefix in the path where the output results will be saved."
             "Three output files will be created:\n"
             "Two with the output positions of cells (with and without orphans)"
             "and one with the number of cells (active and total) per timestamp.\n"
             "- GT File: File with ground truth cell counts (active and total) per timestamp.\n"
             "GT information is used to evaluate the tracker/predictor."
            )

    varLabelFiles.set(crMsg)

    crMsgParam = ("Tracking Parameters:\n\n"
                  "- Threshold (th): distance threshold used to deem a cell matching.\n\n"
                  "Prediction Parameters:\n\n"
                  "- Maximum Error (maxError): maximum error allowed for the prediction.\nWhen"
                  "error > maxError, the prediction model is updated.\n"
                  "- Threshold (th): distance threshold used to deem a cell matching and"
                  "check the distance of predicted points\nto the last observed cell positions.\n"
                  "- Window Size (w): size of the window (number of points) used for predicting"
                  "the next cell positions.\n"
                  "- Renewable w portion (pw): portion of points from the window that will be discarted\n"
                  "to renew the window when the error is more than maxError.\n"
                  "- Polynomial Degree: degree of the polynomial function used for the prediction.\n"
            )
    varLabelParameters.set(crMsgParam)

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

    adjustLabelFormatting()

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import help
    help.vp_start_gui()

def adjustLabelFormatting():
    global w
    w.TLabelFiles.configure(font="-family {helvetica} -size 10 -weight normal")





