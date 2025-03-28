"""
=================================================
@Author: Zenon
@Date: 2025-03-26
@Description：Debugger File
==================================================
"""

from tsadlib import logger

if __name__ == '__main__':
    if 3 % 2 == 0:
        logger.trace('error info')
        logger.debug('error info')
        logger.info('error info')
        logger.success('error info')
        logger.warning('error info')
        logger.error('error info')
        logger.critical('error info')
