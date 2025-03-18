"""
=================================================
@Author: Zenon
@Date: 2025-03-11
@Description：log configuration
==================================================
"""
import os
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger as _logger

from tsadlib.utils.constants import PROJECT_ROOT

# Log levels in loguru (from lowest to highest):
# TRACE (5): Detailed information for debugging
# DEBUG (10): Debugging information
# INFO (20): General information about program execution
# SUCCESS (25): Successful operations
# WARNING (30): Warning messages for potentially problematic situations
# ERROR (40): Error messages for failures that can be handled
# CRITICAL (50): Critical errors that may lead to program termination

# 1️⃣ Gets the current time (for log directory and file name)
current_time = datetime.now()
log_dir = os.path.join(PROJECT_ROOT, "logs", f"{current_time:%Y}", f"{current_time:%m}",
                       f"{current_time:%d}")  # logs/YYYY/mm/dd/
os.makedirs(log_dir, exist_ok=True)  # Automatic directory creation, and ignore exception.

log_file = os.path.join(log_dir, f"{current_time:%H-%M-%S}.log")  # log/YYYY/mm/DD/HH-MM-SS.log

# 2️⃣ Clear the default Loguru configuration (to prevent repeated addition of handlers)
_logger.remove()

# 3️⃣ Configure console logs (INFO and above, color enabled)
_logger.add(
    sink=lambda msg: print(msg, end=""),  # output to console
    level="INFO",  # Will capture INFO (20) and above levels
    colorize=True
)

# 4️⃣ Configuration file logs (all logs are stored in a unique log file)
_logger.add(
    log_file, level="INFO", encoding="utf-8"  # Will capture INFO (20) and above levels
)

# Export the logger with type hint
if TYPE_CHECKING:
    from loguru import Logger

logger: "Logger" = _logger
