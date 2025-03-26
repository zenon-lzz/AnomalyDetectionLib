"""
=================================================
@Author: Zenon
@Date: 2025-03-11
@Description：constants definition
==================================================
"""
import os
import sys

FIX_SEED = 2025

# Project root directory (2 levels up from the current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Check if currently in debug mode
def is_debugging():
    # Method 1: Check if debugger is attached
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is not None and gettrace():
        return True

    # Method 2: Check if running through IDE debugger like PyCharm
    if any(x in sys.modules for x in ['pydevd', 'pdb']):
        return True

    # Method 3: Check command line arguments
    if any(arg in sys.argv for arg in ['-m', 'pdb']):
        return True

    return False


# Current running mode
IS_DEBUG = is_debugging()
