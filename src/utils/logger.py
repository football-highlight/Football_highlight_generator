"""
Logging utilities for the football highlights system
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorlog


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup a logger with consistent formatting
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_project_logger(name: str = None) -> logging.Logger:
    """
    Get a logger configured for the project
    
    Args:
        name: Logger name (defaults to caller module name)
        
    Returns:
        Configured logger
    """
    import inspect
    
    if name is None:
        # Get caller module name
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__ if module else 'unknown'
    
    # Get log level from environment or config
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Get log file from environment or use default
    log_file = os.getenv('LOG_FILE')
    if not log_file:
        # Create logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = logs_dir / f'app_{timestamp}.log'
    
    return setup_logger(
        name=name,
        log_level=log_level,
        log_file=str(log_file),
        console=True
    )


def log_metrics(metrics: dict, logger: logging.Logger, prefix: str = ""):
    """
    Log metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        logger: Logger to use
        prefix: Prefix for log messages
    """
    if prefix:
        prefix = f"{prefix} - "
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Format numbers nicely
            if isinstance(value, float):
                log_str = f"{prefix}{key}: {value:.4f}"
            else:
                log_str = f"{prefix}{key}: {value}"
            logger.info(log_str)
        elif isinstance(value, dict):
            # Recursively log nested dictionaries
            log_metrics(value, logger, f"{prefix}{key}")
        else:
            logger.info(f"{prefix}{key}: {value}")


def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """
    Log an exception with context
    
    Args:
        logger: Logger to use
        exception: Exception to log
        context: Context information
    """
    if context:
        logger.error(f"{context}: {str(exception)}")
    else:
        logger.error(str(exception))
    
    # Log traceback for debugging
    import traceback
    logger.debug(traceback.format_exc())


def create_performance_logger(name: str = "performance") -> logging.Logger:
    """
    Create a logger specifically for performance metrics
    
    Args:
        name: Logger name
        
    Returns:
        Performance logger
    """
    # Create performance logs directory
    perf_dir = Path('logs') / 'performance'
    perf_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = perf_dir / f'performance_{timestamp}.log'
    
    # Create formatter for performance logs
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger


def log_performance(
    operation: str,
    duration: float,
    additional_info: dict = None,
    logger: logging.Logger = None
):
    """
    Log performance metrics
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        additional_info: Additional performance metrics
        logger: Logger to use (creates one if not provided)
    """
    if logger is None:
        logger = create_performance_logger()
    
    # Format the log message
    message = f"PERF - {operation}: {duration:.3f}s"
    
    if additional_info:
        for key, value in additional_info.items():
            if isinstance(value, (int, float)):
                message += f", {key}: {value}"
            else:
                message += f", {key}: {str(value)}"
    
    logger.info(message)


# Example usage
if __name__ == "__main__":
    # Test the logger
    test_logger = setup_logger("test_logger", log_level="DEBUG")
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    # Test metrics logging
    metrics = {
        "accuracy": 0.9234,
        "loss": 0.1234,
        "epoch": 10,
        "details": {
            "train_loss": 0.1,
            "val_loss": 0.15
        }
    }
    
    log_metrics(metrics, test_logger, "Training")
    
    # Test performance logging
    log_performance("video_processing", 2.345, {"frames": 1000, "resolution": "1920x1080"})
    
    print("✅ Logger tests completed successfully!")