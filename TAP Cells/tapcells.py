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

import tapcells_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    tapcells_support.set_Tk_var()
    top = ToplevelTAPCells (root)
    tapcells_support.init(root, top)
    root.mainloop()

w = None
def create_ToplevelTAPCells(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_ToplevelTAPCells(root, *args, **kwargs)' .'''
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    tapcells_support.set_Tk_var()
    top = ToplevelTAPCells (w)
    tapcells_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_ToplevelTAPCells():
    global w
    w.destroy()
    w = None

class ToplevelTAPCells:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("1469x835+379+109")
        top.minsize(1, 1)
        top.maxsize(1470, 862)
        top.resizable(1,  1)
        top.title("Cell Tracking and Prediction")
        top.configure(highlightcolor="black")

        self.btnInputFile = tk.Button(top)
        self.btnInputFile.place(relx=0.606, rely=0.007, height=35, width=50)
        self.btnInputFile.configure(activebackground="#f9f9f9")
        self.btnInputFile.configure(command=tapcells_support.btnSelectInput)
        self.btnInputFile.configure(text='''...''')

        self.lblInputFile = tk.Label(top)
        self.lblInputFile.place(relx=0.088, rely=0.007, height=35, width=759)
        self.lblInputFile.configure(activebackground="#f9f9f9")
        self.lblInputFile.configure(background="#ffffff")
        self.lblInputFile.configure(highlightbackground="#ffffff")
        self.lblInputFile.configure(justify='left')
        self.lblInputFile.configure(text='''Select input file''')
        self.lblInputFile.configure(textvariable=tapcells_support.VarInputFile)

        self.btnOutputFile = tk.Button(top)
        self.btnOutputFile.place(relx=0.606, rely=0.104, height=35, width=50)
        self.btnOutputFile.configure(activebackground="#f9f9f9")
        self.btnOutputFile.configure(command=tapcells_support.btnSelectOutput)
        self.btnOutputFile.configure(text='''...''')

        self.Label2 = tk.Label(top)
        self.Label2.place(relx=0.004, rely=0.019, height=15, width=110)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(text='''Input File:''')

        self.Label2_1 = tk.Label(top)
        self.Label2_1.place(relx=0.007, rely=0.115, height=15, width=110)
        self.Label2_1.configure(activebackground="#f9f9f9")
        self.Label2_1.configure(text='''Output File:''')

        self.lblOutputFile = tk.Label(top)
        self.lblOutputFile.place(relx=0.088, rely=0.103, height=35, width=758)
        self.lblOutputFile.configure(activebackground="#f9f9f9")
        self.lblOutputFile.configure(background="#ffffff")
        self.lblOutputFile.configure(justify='left')
        self.lblOutputFile.configure(text='''Select output file''')
        self.lblOutputFile.configure(textvariable=tapcells_support.VarOutputFile)

        self.lblInputPath = tk.Label(top)
        self.lblInputPath.place(relx=0.088, rely=0.055, height=35, width=759)
        self.lblInputPath.configure(activebackground="#f9f9f9")
        self.lblInputPath.configure(background="#ffffff")
        self.lblInputPath.configure(justify='left')
        self.lblInputPath.configure(text='''Select input path''')
        self.lblInputPath.configure(textvariable=tapcells_support.VarInputPath)

        self.Label2_1_1 = tk.Label(top)
        self.Label2_1_1.place(relx=0.007, rely=0.067, height=15, width=108)
        self.Label2_1_1.configure(activebackground="#f9f9f9")
        self.Label2_1_1.configure(text='''Input Path:''')

        self.btnInputPath = tk.Button(top)
        self.btnInputPath.place(relx=0.606, rely=0.055, height=35, width=50)
        self.btnInputPath.configure(activebackground="#f9f9f9")
        self.btnInputPath.configure(command=tapcells_support.btnSelectPath)
        self.btnInputPath.configure(text='''...''')

        self.TSeparator1 = ttk.Separator(top)
        self.TSeparator1.place(relx=0.013, rely=0.205,  relwidth=0.627)

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.012, rely=0.22, relheight=0.246, relwidth=0.276)

        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")

        self.TComboboxPredictors = ttk.Combobox(self.Frame1)
        self.TComboboxPredictors.place(relx=0.03, rely=0.815, relheight=0.122
                , relwidth=0.953)
        self.value_list = ['CM-Predictor','KDT-Predictor (Lagrange)','BLT-Predictor (Lagrange)','KDT-Predictor (Polynomial)','KDT-Predictor (Weighted Polynomial)','KDT-Predictor (Lagrange) AvgError','KDT-Predictor (Polynomial) AvgError','KDT-Predictor (Weighted Polynomial) AvgError','KDT-Predictor (Lagrange) AngleError','KDT-Predictor (Polynomial) AngleError','KDT-Predictor (Weighted Polynomial) AngleError','Int-Predictor (Lagrange)','Int-Predictor (Polynomial)','Int-Predictor (Weighted Polynomial)',]
        self.TComboboxPredictors.configure(values=self.value_list)
        self.TComboboxPredictors.configure(state='disabled')
        self.TComboboxPredictors.configure(takefocus="")
        self.TComboboxPredictors.bind('<<ComboboxSelected>>',tapcells_support.ComboboxPredictorsSelected)

        self.RadiobuttonPrediction = tk.Radiobutton(self.Frame1)
        self.RadiobuttonPrediction.place(relx=0.005, rely=0.556, relheight=0.083
                , relwidth=0.195)
        self.RadiobuttonPrediction.configure(activebackground="#f9f9f9")
        self.RadiobuttonPrediction.configure(command=tapcells_support.RadioButtonPredSelect)
        self.RadiobuttonPrediction.configure(justify='left')
        self.RadiobuttonPrediction.configure(text='''Predict''')
        self.RadiobuttonPrediction.configure(value=2)
        self.RadiobuttonPrediction.configure(variable=tapcells_support.selectedButton)

        self.TComboboxTrackers = ttk.Combobox(self.Frame1)
        self.TComboboxTrackers.place(relx=0.03, rely=0.307, relheight=0.122
                , relwidth=0.953)
        self.value_list = ['Direct-Tracker','Clever-Tracker','KDT-Tracker','BLT-Tracker',]
        self.TComboboxTrackers.configure(values=self.value_list)
        self.TComboboxTrackers.configure(state='disabled')
        self.TComboboxTrackers.configure(takefocus="")
        self.TComboboxTrackers.bind('<<ComboboxSelected>>',tapcells_support.ComboboxTrackersSelected)

        self.RadiobuttonTracking = tk.Radiobutton(self.Frame1)
        self.RadiobuttonTracking.place(relx=0.007, rely=0.078, relheight=0.083
                , relwidth=0.16)
        self.RadiobuttonTracking.configure(activebackground="#f9f9f9")
        self.RadiobuttonTracking.configure(command=tapcells_support.RadioButtonTrackingSelect)
        self.RadiobuttonTracking.configure(justify='left')
        self.RadiobuttonTracking.configure(text='''Track''')
        self.RadiobuttonTracking.configure(value=1)
        self.RadiobuttonTracking.configure(variable=tapcells_support.selectedButton)

        self.Label5 = tk.Label(self.Frame1)
        self.Label5.place(relx=0.167, rely=0.039, height=45, width=329)
        self.Label5.configure(activebackground="#f9f9f9")
        self.Label5.configure(text='''Tracking algorithm, which reads the seed points of every image and outputs cells trajectories.''')
        self.Label5.configure(wraplength="300")

        self.Label5_1 = tk.Label(self.Frame1)
        self.Label5_1.place(relx=0.2, rely=0.517, height=45, width=319)
        self.Label5_1.configure(activebackground="#f9f9f9")
        self.Label5_1.configure(text='''Prediction algorithm, which reads the seed points of every w images and outputs estimated cells trajectories.''')
        self.Label5_1.configure(wraplength="300")

        self.Labelframe1 = tk.LabelFrame(top)
        self.Labelframe1.place(relx=0.298, rely=0.219, relheight=0.09
                , relwidth=0.149)
        self.Labelframe1.configure(relief='groove')
        self.Labelframe1.configure(text='''Tracking Parameters''')

        self.Label1 = tk.Label(self.Labelframe1)
        self.Label1.place(relx=0.073, rely=0.467, height=15, width=109
                , bordermode='ignore')
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(text='''Threshold (th):''')

        self.EntryTrackerTh = tk.Entry(self.Labelframe1)
        self.EntryTrackerTh.place(relx=0.671, rely=0.4, height=27, relwidth=0.256
                , bordermode='ignore')
        self.EntryTrackerTh.configure(background="white")
        self.EntryTrackerTh.configure(font="TkFixedFont")
        self.EntryTrackerTh.configure(selectbackground="blue")
        self.EntryTrackerTh.configure(selectforeground="white")
        self.EntryTrackerTh.configure(state='disabled')
        self.EntryTrackerTh.configure(textvariable=tapcells_support.varEntryTrackerTh)

        self.LabelframePredictionParam = tk.LabelFrame(top)
        self.LabelframePredictionParam.place(relx=0.455, rely=0.216
                , relheight=0.246, relwidth=0.184)
        self.LabelframePredictionParam.configure(relief='groove')
        self.LabelframePredictionParam.configure(text='''Prediction Parameters''')

        self.Label1_1 = tk.Label(self.LabelframePredictionParam)
        self.Label1_1.place(relx=0.026, rely=0.215, height=15, width=179
                , bordermode='ignore')
        self.Label1_1.configure(activebackground="#f9f9f9")
        self.Label1_1.configure(text='''Maximum Error (maxError):''')

        self.Label3 = tk.Label(self.LabelframePredictionParam)
        self.Label3.place(relx=0.03, rely=0.366, height=15, width=109
                , bordermode='ignore')
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(text='''Threshold (th):''')

        self.Label3_1 = tk.Label(self.LabelframePredictionParam)
        self.Label3_1.place(relx=0.026, rely=0.527, height=15, width=129
                , bordermode='ignore')
        self.Label3_1.configure(activebackground="#f9f9f9")
        self.Label3_1.configure(text='''Window Size (w):''')

        self.Label3_1_1 = tk.Label(self.LabelframePredictionParam)
        self.Label3_1_1.place(relx=0.026, rely=0.683, height=15, width=179
                , bordermode='ignore')
        self.Label3_1_1.configure(activebackground="#f9f9f9")
        self.Label3_1_1.configure(text='''Renewable w portion (pw):''')

        self.Label3_1_1_1 = tk.Label(self.LabelframePredictionParam)
        self.Label3_1_1_1.place(relx=0.041, rely=0.839, height=15, width=131
                , bordermode='ignore')
        self.Label3_1_1_1.configure(activebackground="#f9f9f9")
        self.Label3_1_1_1.configure(justify='left')
        self.Label3_1_1_1.configure(text='''Polynomial Degree:''')

        self.EntryPredMaxError = tk.Entry(self.LabelframePredictionParam)
        self.EntryPredMaxError.place(relx=0.7, rely=0.18, height=27
                , relwidth=0.244, bordermode='ignore')
        self.EntryPredMaxError.configure(background="white")
        self.EntryPredMaxError.configure(font="TkFixedFont")
        self.EntryPredMaxError.configure(selectbackground="blue")
        self.EntryPredMaxError.configure(selectforeground="white")
        self.EntryPredMaxError.configure(state='disabled')
        self.EntryPredMaxError.configure(textvariable=tapcells_support.varEntryPredMaxError)

        self.EntryPredTh = tk.Entry(self.LabelframePredictionParam)
        self.EntryPredTh.place(relx=0.7, rely=0.337, height=27, relwidth=0.244
                , bordermode='ignore')
        self.EntryPredTh.configure(background="white")
        self.EntryPredTh.configure(font="TkFixedFont")
        self.EntryPredTh.configure(selectbackground="blue")
        self.EntryPredTh.configure(selectforeground="white")
        self.EntryPredTh.configure(state='disabled')
        self.EntryPredTh.configure(textvariable=tapcells_support.varEntryPredTh)

        self.EntryPredW = tk.Entry(self.LabelframePredictionParam)
        self.EntryPredW.place(relx=0.7, rely=0.488, height=27, relwidth=0.244
                , bordermode='ignore')
        self.EntryPredW.configure(background="white")
        self.EntryPredW.configure(font="TkFixedFont")
        self.EntryPredW.configure(selectbackground="blue")
        self.EntryPredW.configure(selectforeground="white")
        self.EntryPredW.configure(state='disabled')
        self.EntryPredW.configure(textvariable=tapcells_support.varEntryPredW)

        self.EntryPredPw = tk.Entry(self.LabelframePredictionParam)
        self.EntryPredPw.place(relx=0.7, rely=0.649, height=27, relwidth=0.244
                , bordermode='ignore')
        self.EntryPredPw.configure(background="white")
        self.EntryPredPw.configure(font="TkFixedFont")
        self.EntryPredPw.configure(selectbackground="blue")
        self.EntryPredPw.configure(selectforeground="white")
        self.EntryPredPw.configure(state='disabled')
        self.EntryPredPw.configure(textvariable=tapcells_support.varEntryPredPw)

        self.EntryPredDegree = tk.Entry(self.LabelframePredictionParam)
        self.EntryPredDegree.place(relx=0.7, rely=0.805, height=27
                , relwidth=0.244, bordermode='ignore')
        self.EntryPredDegree.configure(background="white")
        self.EntryPredDegree.configure(font="TkFixedFont")
        self.EntryPredDegree.configure(selectbackground="blue")
        self.EntryPredDegree.configure(selectforeground="white")
        self.EntryPredDegree.configure(state='disabled')
        self.EntryPredDegree.configure(textvariable=tapcells_support.varEntryPredDegree)

        self.Labelframe2 = tk.LabelFrame(top)
        self.Labelframe2.place(relx=0.014, rely=0.527, relheight=0.461
                , relwidth=0.626)
        self.Labelframe2.configure(relief='groove')
        self.Labelframe2.configure(text='''Data Information and Results''')

        self.TSeparator2 = ttk.Separator(self.Labelframe2)
        self.TSeparator2.place(relx=0.37, rely=0.039, relheight=0.935
                , bordermode='ignore')
        self.TSeparator2.configure(orient="vertical")

        self.LabelCommand = tk.Label(self.Labelframe2)
        self.LabelCommand.place(relx=0.011, rely=0.052, height=115, width=319
                , bordermode='ignore')
        self.LabelCommand.configure(activebackground="#f9f9f9")
        self.LabelCommand.configure(relief="groove")
        self.LabelCommand.configure(textvariable=tapcells_support.varLabelCommand)
        self.LabelCommand.configure(wraplength="310")
        self.tooltip_font = "TkDefaultFont"
        self.LabelCommand_tooltip = \
        ToolTip(self.LabelCommand, self.tooltip_font, '''Executed command''')

        self.Label4 = tk.Label(self.Labelframe2)
        self.Label4.place(relx=0.011, rely=0.558, height=16, width=123
                , bordermode='ignore')
        self.Label4.configure(activebackground="#f9f9f9")
        self.Label4.configure(justify='left')
        self.Label4.configure(text='''Sequence Size:''')

        self.LabelSequenceSize = tk.Label(self.Labelframe2)
        self.LabelSequenceSize.place(relx=0.283, rely=0.558, height=15, width=72
                , bordermode='ignore')
        self.LabelSequenceSize.configure(activebackground="#f9f9f9")
        self.LabelSequenceSize.configure(text='''--''')
        self.LabelSequenceSize.configure(textvariable=tapcells_support.varResultSeqSize)

        self.Label4_1 = tk.Label(self.Labelframe2)
        self.Label4_1.place(relx=0.009, rely=0.831, height=15, width=238
                , bordermode='ignore')
        self.Label4_1.configure(activebackground="#f9f9f9")
        self.Label4_1.configure(justify='left')
        self.Label4_1.configure(text='''Tracking Detection Rate (TDR):''')

        self.LabelTDR = tk.Label(self.Labelframe2)
        self.LabelTDR.place(relx=0.283, rely=0.831, height=15, width=72
                , bordermode='ignore')
        self.LabelTDR.configure(activebackground="#f9f9f9")
        self.LabelTDR.configure(text='''--''')
        self.LabelTDR.configure(textvariable=tapcells_support.varResultTDR)

        self.Label4_1_1 = tk.Label(self.Labelframe2)
        self.Label4_1_1.place(relx=0.011, rely=0.925, height=15, width=264
                , bordermode='ignore')
        self.Label4_1_1.configure(activebackground="#f9f9f9")
        self.Label4_1_1.configure(justify='left')
        self.Label4_1_1.configure(text='''Total Cells Relative Error (TCRE):''')

        self.LabelTCRE = tk.Label(self.Labelframe2)
        self.LabelTCRE.place(relx=0.283, rely=0.922, height=15, width=72
                , bordermode='ignore')
        self.LabelTCRE.configure(activebackground="#f9f9f9")
        self.LabelTCRE.configure(text='''--''')
        self.LabelTCRE.configure(textvariable=tapcells_support.varResultTCRE)

        self.Label4_1_1_1 = tk.Label(self.Labelframe2)
        self.Label4_1_1_1.place(relx=0.009, rely=0.74, height=15, width=112
                , bordermode='ignore')
        self.Label4_1_1_1.configure(activebackground="#f9f9f9")
        self.Label4_1_1_1.configure(justify='left')
        self.Label4_1_1_1.configure(text='''Total Cells:''')

        self.LabelTotalCells = tk.Label(self.Labelframe2)
        self.LabelTotalCells.place(relx=0.283, rely=0.74, height=15, width=72
                , bordermode='ignore')
        self.LabelTotalCells.configure(activebackground="#f9f9f9")
        self.LabelTotalCells.configure(text='''--''')
        self.LabelTotalCells.configure(textvariable=tapcells_support.varResultTotalCells)

        self.Label4_1_1_1_1 = tk.Label(self.Labelframe2)
        self.Label4_1_1_1_1.place(relx=0.011, rely=0.649, height=15, width=212
                , bordermode='ignore')
        self.Label4_1_1_1_1.configure(activebackground="#f9f9f9")
        self.Label4_1_1_1_1.configure(justify='left')
        self.Label4_1_1_1_1.configure(text='''Total Cells (Ground Truth):''')

        self.LabelTotalCellsGT = tk.Label(self.Labelframe2)
        self.LabelTotalCellsGT.place(relx=0.283, rely=0.649, height=15, width=72
                , bordermode='ignore')
        self.LabelTotalCellsGT.configure(activebackground="#f9f9f9")
        self.LabelTotalCellsGT.configure(text='''--''')
        self.LabelTotalCellsGT.configure(textvariable=tapcells_support.varTotalCellsGT)

        self.TFrameTrajectoryPlot = ttk.Frame(self.Labelframe2)
        self.TFrameTrajectoryPlot.place(relx=0.378, rely=0.034, relheight=0.948
                , relwidth=0.614, bordermode='ignore')
        self.TFrameTrajectoryPlot.configure(relief='flat')
        self.TFrameTrajectoryPlot.configure(borderwidth="1")
        self.TFrameTrajectoryPlot.configure(relief="flat")

        self.Label4_2 = tk.Label(self.Labelframe2)
        self.Label4_2.place(relx=0.011, rely=0.468, height=16, width=203
                , bordermode='ignore')
        self.Label4_2.configure(activebackground="#f9f9f9")
        self.Label4_2.configure(justify='left')
        self.Label4_2.configure(text='''Execution time (with I/O):''')

        self.LabelExecutionTime = tk.Label(self.Labelframe2)
        self.LabelExecutionTime.place(relx=0.283, rely=0.468, height=15, width=72
                , bordermode='ignore')
        self.LabelExecutionTime.configure(activebackground="#f9f9f9")
        self.LabelExecutionTime.configure(text='''--''')
        self.LabelExecutionTime.configure(textvariable=tapcells_support.varResultExecTime)

        self.Label4_2_1 = tk.Label(self.Labelframe2)
        self.Label4_2_1.place(relx=0.011, rely=0.39, height=16, width=83
                , bordermode='ignore')
        self.Label4_2_1.configure(activebackground="#f9f9f9")
        self.Label4_2_1.configure(font="-family {helvetica} -size 8 -weight bold")
        self.Label4_2_1.configure(text='''Statistics:''')

        self.ButtonRun = tk.Button(top)
        self.ButtonRun.place(relx=0.014, rely=0.478, height=35, width=919)
        self.ButtonRun.configure(activebackground="#f9f9f9")
        self.ButtonRun.configure(command=tapcells_support.BtnRunAlgorithmClick)
        self.ButtonRun.configure(state='active')
        self.ButtonRun.configure(text='''Run Algorithm (the execution may take a while)''')

        self.Label2_1_2 = tk.Label(top)
        self.Label2_1_2.place(relx=0.01, rely=0.163, height=15, width=80)
        self.Label2_1_2.configure(activebackground="#f9f9f9")
        self.Label2_1_2.configure(text='''GT File:''')

        self.lblGTFile = tk.Label(top)
        self.lblGTFile.place(relx=0.088, rely=0.151, height=35, width=758)
        self.lblGTFile.configure(activebackground="#f9f9f9")
        self.lblGTFile.configure(background="#ffffff")
        self.lblGTFile.configure(justify='left')
        self.lblGTFile.configure(text='''Select output file''')
        self.lblGTFile.configure(textvariable=tapcells_support.VarGTFile)

        self.btnGTFile = tk.Button(top)
        self.btnGTFile.place(relx=0.606, rely=0.151, height=35, width=50)
        self.btnGTFile.configure(activebackground="#f9f9f9")
        self.btnGTFile.configure(command=tapcells_support.btnSelectGTFile)
        self.btnGTFile.configure(text='''...''')

        self.Label5_2 = tk.Label(top)
        self.Label5_2.place(relx=0.297, rely=0.407, height=45, width=229)
        self.Label5_2.configure(activebackground="#f9f9f9")
        self.Label5_2.configure(text='''For a complete description of the required parameters, please refer to the Help menu.''')
        self.Label5_2.configure(wraplength="220")

        self.TFrameCellPositions = ttk.Frame(top)
        self.TFrameCellPositions.place(relx=0.647, rely=0.012, relheight=0.449
                , relwidth=0.344)
        self.TFrameCellPositions.configure(relief='flat')
        self.TFrameCellPositions.configure(borderwidth="2")
        self.TFrameCellPositions.configure(relief="flat")

        self.TFrameOutputs = ttk.Frame(top)
        self.TFrameOutputs.place(relx=0.647, rely=0.539, relheight=0.453
                , relwidth=0.169)
        self.TFrameOutputs.configure(relief='flat')
        self.TFrameOutputs.configure(borderwidth="2")
        self.TFrameOutputs.configure(relief="flat")

        self.TFrameGroundTruth = ttk.Frame(top)
        self.TFrameGroundTruth.place(relx=0.822, rely=0.535, relheight=0.453
                , relwidth=0.169)
        self.TFrameGroundTruth.configure(relief='flat')
        self.TFrameGroundTruth.configure(borderwidth="2")
        self.TFrameGroundTruth.configure(relief="flat")

        self.menubar = tk.Menu(top,font="TkMenuFont",bg=_bgcolor,fg=_fgcolor)
        top.configure(menu = self.menubar)

        self.menubar.add_command(
                command=tapcells_support.menuHelpClick,
                label="Help")
        self.menubar.add_command(
                command=tapcells_support.menuAboutClick,
                label="About")
        self.menubar.add_command(
                command=tapcells_support.menuExitClick,
                label="Exit")

        self.LabelRedInformation = tk.Label(top)
        self.LabelRedInformation.place(relx=0.298, rely=0.311, height=66
                , width=228)
        self.LabelRedInformation.configure(activebackground="#f9f9f9")
        self.LabelRedInformation.configure(foreground="#c40c0c")
        self.LabelRedInformation.configure(text='''Please, inform the task (Track or Predict).''')
        self.LabelRedInformation.configure(textvariable=tapcells_support.varRedInformation)
        self.LabelRedInformation.configure(wraplength="220")

        self.LabelOutputResultsTable = tk.Label(top)
        self.LabelOutputResultsTable.place(relx=0.647, rely=0.491, height=16
                , width=236)
        self.LabelOutputResultsTable.configure(cursor="fleur")
        self.LabelOutputResultsTable.configure(text='''Label''')
        self.LabelOutputResultsTable.configure(textvariable=tapcells_support.varLabelOutputResultsTable)

        self.LabelOutputGTTable = tk.Label(top)
        self.LabelOutputGTTable.place(relx=0.824, rely=0.491, height=16
                , width=216)
        self.LabelOutputGTTable.configure(activebackground="#f9f9f9")
        self.LabelOutputGTTable.configure(text='''Cells per timestamp (from Ground Truth):''')
        self.LabelOutputGTTable.configure(textvariable=tapcells_support.varLabelOutputGTTable)

# ======================================================
# Support code for Balloon Help (also called tooltips).
# Found the original code at:
# http://code.activestate.com/recipes/576688-tooltip-for-tkinter/
# Modified by Rozen to remove Tkinter import statements and to receive
# the font as an argument.
# ======================================================

from time import time, localtime, strftime

class ToolTip(tk.Toplevel):
    """
    Provides a ToolTip widget for Tkinter.
    To apply a ToolTip to any Tkinter widget, simply pass the widget to the
    ToolTip constructor
    """
    def __init__(self, wdgt, tooltip_font, msg=None, msgFunc=None,
                 delay=0.5, follow=True):
        """
        Initialize the ToolTip

        Arguments:
          wdgt: The widget this ToolTip is assigned to
          tooltip_font: Font to be used
          msg:  A static string message assigned to the ToolTip
          msgFunc: A function that retrieves a string to use as the ToolTip text
          delay:   The delay in seconds before the ToolTip appears(may be float)
          follow:  If True, the ToolTip follows motion, otherwise hides
        """
        self.wdgt = wdgt
        # The parent of the ToolTip is the parent of the ToolTips widget
        self.parent = self.wdgt.master
        # Initalise the Toplevel
        tk.Toplevel.__init__(self, self.parent, bg='black', padx=1, pady=1)
        # Hide initially
        self.withdraw()
        # The ToolTip Toplevel should have no frame or title bar
        self.overrideredirect(True)

        # The msgVar will contain the text displayed by the ToolTip
        self.msgVar = tk.StringVar()
        if msg is None:
            self.msgVar.set('No message provided')
        else:
            self.msgVar.set(msg)
        self.msgFunc = msgFunc
        self.delay = delay
        self.follow = follow
        self.visible = 0
        self.lastMotion = 0
        # The text of the ToolTip is displayed in a Message widget
        tk.Message(self, textvariable=self.msgVar, bg='#FFFFDD',
                font=tooltip_font,
                aspect=1000).grid()

        # Add bindings to the widget.  This will NOT override
        # bindings that the widget already has
        self.wdgt.bind('<Enter>', self.spawn, '+')
        self.wdgt.bind('<Leave>', self.hide, '+')
        self.wdgt.bind('<Motion>', self.move, '+')

    def spawn(self, event=None):
        """
        Spawn the ToolTip.  This simply makes the ToolTip eligible for display.
        Usually this is caused by entering the widget

        Arguments:
          event: The event that called this funciton
        """
        self.visible = 1
        # The after function takes a time argument in milliseconds
        self.after(int(self.delay * 1000), self.show)

    def show(self):
        """
        Displays the ToolTip if the time delay has been long enough
        """
        if self.visible == 1 and time() - self.lastMotion > self.delay:
            self.visible = 2
        if self.visible == 2:
            self.deiconify()

    def move(self, event):
        """
        Processes motion within the widget.
        Arguments:
          event: The event that called this function
        """
        self.lastMotion = time()
        # If the follow flag is not set, motion within the
        # widget will make the ToolTip disappear
        #
        if self.follow is False:
            self.withdraw()
            self.visible = 1

        # Offset the ToolTip 10x10 pixes southwest of the pointer
        self.geometry('+%i+%i' % (event.x_root+20, event.y_root-10))
        try:
            # Try to call the message function.  Will not change
            # the message if the message function is None or
            # the message function fails
            self.msgVar.set(self.msgFunc())
        except:
            pass
        self.after(int(self.delay * 1000), self.show)

    def hide(self, event=None):
        """
        Hides the ToolTip.  Usually this is caused by leaving the widget
        Arguments:
          event: The event that called this function
        """
        self.visible = 0
        self.withdraw()

    def update(self, msg):
        """
        Updates the Tooltip with a new message. Added by Rozen
        """
        self.msgVar.set(msg)

# ===========================================================
#                   End of Class ToolTip
# ===========================================================

if __name__ == '__main__':
    vp_start_gui()





