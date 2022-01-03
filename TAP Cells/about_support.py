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
    global varCopyrightLabel
    varCopyrightLabel = tk.StringVar()

    crMsg = ("TAP Cells: Tracking and predicting cell positions from sequences\n" +
            "of microscopic images depicting developing embryos.\n" +
            "Copyright (C) 2021  Mirela Cazzolato\n\n" +
            "This program is free software: you can redistribute it and/or modify\n" +
            "it under the terms of the GNU General Public License as published by\n" +
            "the Free Software Foundation, either version 3 of the License, or\n" +
            "(at your option) any later version.\n" +
            "This program is distributed in the hope that it will be useful,\n" +
            "but WITHOUT ANY WARRANTY; without even the implied warranty of\n" +
            "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n" +
            "GNU General Public License for more details.\n" +
            "You should have received a copy of the GNU General Public License\n" +
            "along with this program.  If not, see <https://www.gnu.org/licenses/>.\n")

    varCopyrightLabel.set(crMsg)

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
    import about
    about.vp_start_gui()

def adjustLabelFormatting():
    global w
    w.TLabelAboutText.configure(font="-family {helvetica} -size 10 -weight bold")
    w.TLabelCopyrightText.configure(font="-family {helvetica} -size 10")
