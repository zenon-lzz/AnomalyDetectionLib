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

_global_logger_config: Dict[str, Any] = {
    "log_dir_base": None,  # Default to LIBRARY_ROOT/logs
    "log_level": "INFO",
    "comment": ""
}


class LoggerSetup:
    """
    A class to set up and manage a Loguru logger instance.
    Ensures that the logger is configured only once and shared.
    Configuration can be set globally before the first logger access.
    """
    _logger_instance: Optional['_LoggerType'] = None

    @staticmethod  # Changed to staticmethod as it operates on class-level instance
    def _setup_logger_instance(log_dir_base: Optional[str], log_level: str, comment: str = ''):
        """
        Configures the Loguru logger with console and file sinks.
        This method actually creates and configures the logger handlers.
        """
        current_time = datetime.now()
        if log_dir_base is None:
            log_dir_base = os.path.join(LIBRARY_ROOT, "logs")

        log_dir = os.path.join(log_dir_base, f"{current_time:%Y}", f"{current_time:%m}",
                               f"{current_time:%d}")
        os.makedirs(log_dir, exist_ok=True)

        log_file_name = f"{current_time:%H-%M-%S}{comment}.log"
        log_file = os.path.join(log_dir, log_file_name)

        _loguru_logger_instance.remove()  # Clear any existing default handlers

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
        LoggerSetup._logger_instance = _loguru_logger_instance

    @staticmethod  # Changed to staticmethod
    def get_logger() -> '_LoggerType':
        """
        Returns the configured logger instance.
        If not already configured, it uses the global configuration or defaults.
        """
        if LoggerSetup._logger_instance is None:
            config = _global_logger_config
            LoggerSetup._setup_logger_instance(
                log_dir_base=config["log_dir_base"],
                log_level=config["log_level"],
                comment=config["comment"]
            )
        return LoggerSetup._logger_instance  # type: ignore[return-value]


def configure_global_logger(log_dir_base: Optional[str] = None, log_level: Optional[str] = None,
                            comment: Optional[str] = None) -> '_LoggerType':
    """
    Sets the global configuration for the logger. 
    This function should be called BEFORE the first call to `get_logger` or before importing `logger` 
    if you want to override the default settings.

    Args:
        log_dir_base (Optional[str]): Base directory for logs. Defaults to LIBRARY_ROOT/logs.
        log_level (Optional[str]): Minimum log level. Defaults to 'INFO'.
        comment (Optional[str]): Suffix for the log file name. Defaults to ''.
    """
    if LoggerSetup._logger_instance is not None:
        # Optionally, log a warning or raise an error if logger is already initialized
        _loguru_logger_instance.warning(
            "Logger already initialized. Configuration changes might not apply to existing handlers or may re-initialize.")
        # To be safe, let's re-initialize if called after first use, though this might have side effects
        # For a stricter approach, one might raise an error here.
        LoggerSetup._logger_instance = None  # Force re-initialization on next get_logger call

    if log_dir_base is not None:
        _global_logger_config["log_dir_base"] = log_dir_base
    if log_level is not None:
        _global_logger_config["log_level"] = log_level
    if comment is not None:
        _global_logger_config["comment"] = comment

    return LoggerSetup.get_logger()


# Export the logger instance directly. It will be configured on first use.
log: '_LoggerType' = LoggerSetup.get_logger()

if __name__ == '__main__':
    # Default logger usage
    log.info("This is an info message using default config.")

    # Simulate another module configuring the logger before use
    # In a real scenario, this would be in a different file, called before `logger` is first used.
    print("\nConfiguring logger with custom settings...")
    log = configure_global_logger(log_level="DEBUG", comment="_custom_run")

    log.debug(
        "This is a debug message using custom config. Log file will be in custom_logs and have _custom_run suffix.")
    log.info("Info message from custom_logger.")

    # If another module just imports `logger` after configuration:
    # from tsadlib.configs.log_config import logger as another_logger
    # another_logger.info("This message should use the custom config if configure_global_logger was called before this import.")
