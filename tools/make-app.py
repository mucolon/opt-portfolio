#!/usr/local/bin/python3
from pyshortcuts import make_shortcut
import os

cwd = "/Users/marcoucolon/Documents/GitHub/opt-portfolio/tools"
cwd_parent = os.path.dirname(cwd)
path_icon = cwd + "/portfolio-analysis.icns"
path_script = cwd_parent + "/analysis.py"
name_app = "Portfolio Analysis"
executable = "/usr/local/bin/python3"

make_shortcut(path_script, name=name_app, icon=path_icon,
              folder=cwd_parent, terminal=True, executable=executable)
