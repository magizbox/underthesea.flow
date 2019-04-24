# -*- coding: utf-8 -*-
import os

__author__ = """Vu Anh"""
__email__ = 'brother.rain.1024@gmail.com'

###########################################################
# Metadata
###########################################################

# Version
try:
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_file, 'r') as infile:
        __version__ = infile.read().strip()
except NameError:
    __version__ = 'unknown (running code interactively?)'
except IOError as ex:
    __version__ = "unknown (%s)" % ex
