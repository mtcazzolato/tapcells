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

import about_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    about_support.set_Tk_var()
    top = ToplevelAbout (root)
    about_support.init(root, top)
    root.mainloop()

w = None
def create_ToplevelAbout(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_ToplevelAbout(root, *args, **kwargs)' .'''
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    about_support.set_Tk_var()
    top = ToplevelAbout (w)
    about_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_ToplevelAbout():
    global w
    w.destroy()
    w = None

class ToplevelAbout:
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

        top.geometry("517x504+1190+275")
        top.minsize(0, 0)
        top.maxsize(2545, 1050)
        top.resizable(1,  1)
        top.title("About TAP Cells")
        top.configure(highlightcolor="black")

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.019, rely=0.02, relheight=0.954, relwidth=0.961)

        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(cursor="fleur")

        self.TLabel1 = ttk.Label(self.Frame1)
        self.TLabel1.place(relx=0.018, rely=0.015, height=32, width=482)
        self.TLabel1.configure(background="#d9d9d9")
        self.TLabel1.configure(foreground="#3700ed")
        self.TLabel1.configure(font="-family {helvetica} -size 11 -weight bold")
        self.TLabel1.configure(relief="flat")
        self.TLabel1.configure(anchor='w')
        self.TLabel1.configure(justify='center')
        self.TLabel1.configure(wraplength="400")
        self.TLabel1.configure(text='''TAP Cells: Tracking and predicting cell positions''')
        self.TLabel1.configure(cursor="fleur")

        self.TLabelAboutText = ttk.Label(self.Frame1)
        self.TLabelAboutText.place(relx=0.02, rely=0.121, height=107, width=471)
        self.TLabelAboutText.configure(background="#d9d9d9")
        self.TLabelAboutText.configure(foreground="#000000")
        self.TLabelAboutText.configure(font="-family {helvetica} -size 10 -weight bold")
        self.TLabelAboutText.configure(relief="flat")
        self.TLabelAboutText.configure(anchor='w')
        self.TLabelAboutText.configure(justify='center')
        self.TLabelAboutText.configure(wraplength="470")
        self.TLabelAboutText.configure(text='''TAP Cells is a prototype that implements the algorithms reported in [Cazzolato et al, 2022]:\n\n"Establishing trajectories of moving objects without identities: The intricacies of cell tracking and a solution" - Journal Information Systems.\n\nFor further information, please access the repository: https://github.com/mtcazzolato/tapcells.\n\n This software is intended for research purposes only.''')
        self.TLabelAboutText.configure(cursor="fleur")

        self.TLabelCopyrightText = ttk.Label(self.Frame1)
        self.TLabelCopyrightText.place(relx=0.04, rely=0.432, height=262
                , width=466)
        self.TLabelCopyrightText.configure(background="#d9d9d9")
        self.TLabelCopyrightText.configure(foreground="#000000")
        self.TLabelCopyrightText.configure(font="-family {helvetica} -size 8")
        self.TLabelCopyrightText.configure(borderwidth="2")
        self.TLabelCopyrightText.configure(relief="flat")
        self.TLabelCopyrightText.configure(anchor='center')
        self.TLabelCopyrightText.configure(justify='left')
        self.TLabelCopyrightText.configure(textvariable=about_support.varCopyrightLabel)

if __name__ == '__main__':
    vp_start_gui()





