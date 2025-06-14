"""
=================================================
@Author: Zenon
@Date: 2025-03-11
@Description：constants definition
==================================================
"""
import datetime
import os

# =================================================
# Global Constants
# =================================================
FIX_SEED = datetime.datetime.now().year  # Dynamically get current year as seed

# =================================================
# Dataset Constants
# Project root directory (2 levels up from the current file)
LIBRARY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The types of tasks supported by tasdlib
TASK_OPTIONS = [
    'benchmarks',
    'tuning'
]
# The types of datasets supported by tasdlib
DATASET_OPTIONS = [
    'MSL',
    'SMAP',
    'PSM',
    'SMD',
    'SwaT',
    'WADI',
    'NIPS_TS_Water',
    'NIPS_TS_Swan',
    'NIPS_TS_CCard'
]
