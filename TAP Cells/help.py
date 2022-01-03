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

import help_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    help_support.set_Tk_var()
    top = ToplevelHelp (root)
    help_support.init(root, top)
    root.mainloop()

w = None
def create_ToplevelHelp(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_ToplevelHelp(root, *args, **kwargs)' .'''
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    help_support.set_Tk_var()
    top = ToplevelHelp (w)
    help_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_ToplevelHelp():
    global w
    w.destroy()
    w = None

class ToplevelHelp:
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

        top.geometry("700x499+1011+250")
        top.minsize(1, 1)
        top.maxsize(2545, 1050)
        top.resizable(1,  1)
        top.title("TAP Cells: Help")

        self.TFrame1 = ttk.Frame(top)
        self.TFrame1.place(relx=0.014, rely=0.022, relheight=0.944
                , relwidth=0.976)
        self.TFrame1.configure(relief='groove')
        self.TFrame1.configure(borderwidth="2")
        self.TFrame1.configure(relief="groove")
        self.TFrame1.configure(cursor="fleur")

        self.TLabelFiles = ttk.Label(self.TFrame1)
        self.TLabelFiles.place(relx=0.015, rely=0.076, height=118, width=662)
        self.TLabelFiles.configure(background="#d9d9d9")
        self.TLabelFiles.configure(foreground="#000000")
        self.TLabelFiles.configure(font="TkDefaultFont")
        self.TLabelFiles.configure(relief="flat")
        self.TLabelFiles.configure(anchor='w')
        self.TLabelFiles.configure(justify='left')
        self.TLabelFiles.configure(wraplength="689")
        self.TLabelFiles.configure(textvariable=help_support.varLabelFiles)

        self.TLabelParameters = ttk.Label(self.TFrame1)
        self.TLabelParameters.place(relx=0.015, rely=0.406, height=237
                , width=662)
        self.TLabelParameters.configure(background="#d9d9d9")
        self.TLabelParameters.configure(foreground="#000000")
        self.TLabelParameters.configure(relief="flat")
        self.TLabelParameters.configure(anchor='w')
        self.TLabelParameters.configure(justify='left')
        self.TLabelParameters.configure(textvariable=help_support.varLabelParameters)

        self.TLabel1 = ttk.Label(self.TFrame1)
        self.TLabel1.place(relx=0.029, rely=0.021, height=16, width=367)
        self.TLabel1.configure(background="#d9d9d9")
        self.TLabel1.configure(foreground="#2414ff")
        self.TLabel1.configure(font="-family {helvetica} -size 8 -weight bold")
        self.TLabel1.configure(relief="flat")
        self.TLabel1.configure(anchor='w')
        self.TLabel1.configure(justify='center')
        self.TLabel1.configure(text='''1. Select input and output files:''')
        self.TLabel1.configure(compound='center')
        self.TLabel1.configure(cursor="fleur")

        self.TLabel1_1 = ttk.Label(self.TFrame1)
        self.TLabel1_1.place(relx=0.029, rely=0.352, height=16, width=367)
        self.TLabel1_1.configure(background="#d9d9d9")
        self.TLabel1_1.configure(foreground="#2414ff")
        self.TLabel1_1.configure(font="-family {helvetica} -size 10 -weight bold")
        self.TLabel1_1.configure(relief="flat")
        self.TLabel1_1.configure(anchor='w')
        self.TLabel1_1.configure(justify='center')
        self.TLabel1_1.configure(text='''2. Inform the corresponding parameters:''')
        self.TLabel1_1.configure(compound='center')

        self.TLabel1_1_1 = ttk.Label(self.TFrame1)
        self.TLabel1_1_1.place(relx=0.029, rely=0.934, height=16, width=647)
        self.TLabel1_1_1.configure(background="#d9d9d9")
        self.TLabel1_1_1.configure(foreground="#2414ff")
        self.TLabel1_1_1.configure(font="-family {helvetica} -size 10 -weight bold")
        self.TLabel1_1_1.configure(relief="flat")
        self.TLabel1_1_1.configure(anchor='w')
        self.TLabel1_1_1.configure(justify='center')
        self.TLabel1_1_1.configure(text='''3. Click on 'Run Algorithm' (the execution may take a while, depending on the data size and approach).''')
        self.TLabel1_1_1.configure(compound='center')

if __name__ == '__main__':
    vp_start_gui()





