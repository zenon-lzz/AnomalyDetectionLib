"""
=================================================
@Author: Zenon
@Date: 2025-03-11
@Description：
==================================================
"""
from .data_provider.data_factory import data_provider
from .models.TimesNet import Model as TimesNet
from .utils.logger import logger
from .utils.traning_stoper import EarlyStopping
from .utils.type import ConfigType

__version__ = '0.1.0'
__all__ = [
    'logger',
    'ConfigType',
    'data_provider',
    'TimesNet',
    'EarlyStopping'
]
