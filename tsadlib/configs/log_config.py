"""
=================================================
@Author: Zenon
@Date: 2025-03-11
@Description：log configuration
==================================================
"""
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Dict, Any

from loguru import logger as _loguru_logger_instance

from tsadlib.configs.constants import LIBRARY_ROOT

if TYPE_CHECKING:
    from loguru import Logger as _LoggerType

# Keep _loguru_logger_instance as the actual Loguru instance we are configuring
# This avoids confusion with LoggerSetup._logger_instance which is our flag

_global_logger_config: Dict[str, Any] = {
    "log_dir_base": None,  # Default to LIBRARY_ROOT/logs
    "log_level": "INFO",
    "comment": ""
}


class LoggerSetup:
    """
    A class to set up and manage a Loguru logger instance.
    Ensures that the logger is configured only once unless reconfigured explicitly.
    """
    _is_configured: bool = False  # Flag to check if initial configuration has happened

    @staticmethod
    def _setup_logger_instance(log_dir_base: Optional[str], log_level: str, comment: str = ''):
        """
        Configures the Loguru logger with console and file sinks.
        This method actually creates and configures the logger handlers.
        It will remove existing handlers before adding new ones.
        """
        current_time = datetime.now()
        if log_dir_base is None:
            log_dir_base = os.path.join(LIBRARY_ROOT, "logs")

        log_dir = os.path.join(log_dir_base, f"{current_time:%Y}", f"{current_time:%m}",
                               f"{current_time:%d}")
        os.makedirs(log_dir, exist_ok=True)

        log_file_name = f"{current_time:%H-%M-%S}{comment}.log"
        log_file = os.path.join(log_dir, log_file_name)

        # Remove all previously added handlers from the global loguru logger instance
        # This is crucial for reconfiguration to work cleanly.
        _loguru_logger_instance.remove()

        _loguru_logger_instance.add(
            sink=lambda msg: print(msg, end=""),
            level=log_level,
            colorize=True,
            format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {name}:{function}:{line} | <level>{level}</level>: <level>{message}</level>'
        )

        _loguru_logger_instance.add(
            log_file,
            level=log_level,
            encoding="utf-8",
            format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {name}:{function}:{line} | <level>{level}</level>: <level>{message}</level>'
        )
        LoggerSetup._is_configured = True

    @staticmethod
    def get_logger() -> '_LoggerType':
        """
        Returns the configured logger instance.
        If not already configured, it uses the global configuration or defaults.
        """
        if not LoggerSetup._is_configured:
            # If not configured, setup with current global_logger_config
            config = _global_logger_config
            LoggerSetup._setup_logger_instance(
                log_dir_base=config["log_dir_base"],
                log_level=config["log_level"],
                comment=config["comment"]
            )
        return _loguru_logger_instance  # Always return the globally managed instance


def configure_global_logger(
        log_dir_base: Optional[str] = None,
        log_level: Optional[str] = None,
        comment: Optional[str] = None
) -> '_LoggerType':
    """
    Sets the global configuration and re-configures the logger.
    This function can be called at any time to change logger settings.

    Args:
        log_dir_base (Optional[str]): Base directory for logs. Defaults to LIBRARY_ROOT/logs if None.
        log_level (Optional[str]): Minimum log level. Defaults to 'INFO' if None.
        comment (Optional[str]): Suffix for the log file name. Defaults to '' if None.
    """
    # Update global config dictionary with provided values if they are not None
    if log_dir_base is not None:
        _global_logger_config["log_dir_base"] = log_dir_base
    if log_level is not None:
        _global_logger_config["log_level"] = log_level
    if comment is not None:
        _global_logger_config["comment"] = comment

    # Apply the new configuration (or existing if no new values were passed for some items)
    # This will remove old handlers and add new ones.
    LoggerSetup._setup_logger_instance(
        log_dir_base=_global_logger_config["log_dir_base"],
        log_level=_global_logger_config["log_level"],
        comment=_global_logger_config["comment"]
    )
    return _loguru_logger_instance


# Export the logger instance directly. 
# It will be configured with defaults when get_logger() is first called (i.e., upon module import).
# Subsequent calls to configure_global_logger() will then re-configure it.
log: '_LoggerType' = LoggerSetup.get_logger()

if __name__ == '__main__':
    # log is already initialized with default settings due to the line above
    log.info(f"Initial log (default settings). Log file will be in default location.")

    print("\nReconfiguring logger with custom settings...")
    # This call will now properly reconfigure the logger by removing old handlers
    # and adding new ones based on the updated _global_logger_config.
    custom_log = configure_global_logger(log_dir_base=os.path.join(LIBRARY_ROOT, "custom_logs"), log_level="DEBUG",
                                         comment="_custom_run")

    custom_log.debug(
        "This is a debug message using custom config. Log file will be in custom_logs and have _custom_run suffix.")
    custom_log.info("Info message from custom_logger.")

    # If another module just imports `log` from this module AFTER configure_global_logger has been called,
    # it will get the reconfigured logger instance.
    # For example:
    # from tsadlib.configs.log_config import log as another_logger_instance
    # another_logger_instance.info("This message uses the reconfigured settings.")

    # To demonstrate that the original 'log' variable also reflects the change:
    log.error("This error message from the original 'log' variable also uses the new custom config.")
