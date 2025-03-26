#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File (Python):  'magic_utils.py'
author:         Julien Straubhaar
date:           2024

Define new magic command (for IPython).
"""

# ------------------------------------------------------------------------------
# define %%skip_if magic command (from https://kioku-space.com/en/jupyter-skip-execution/)
# usage: %%skip_if <expression>

from IPython.core.magic import register_cell_magic
from IPython import get_ipython

@register_cell_magic
def skip_if(line, cell):
    if eval(line):
        return
    get_ipython().run_cell(cell)
# ------------------------------------------------------------------------------
