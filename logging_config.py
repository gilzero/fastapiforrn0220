# filepath: logging_config.py
import logging
import logging.handlers
from pathlib import Path
import sys
import json
from datetime import datetime, UTC


class CustomFormatter(logging.Formatter):
    """Custom formatter that includes more context for debug logs"""
    
    def format(self, record):
        # Add timezone-aware UTC timestamp
        record.timestamp = datetime.now(UTC).isoformat()
        
        # For debug logs, add extra context if available
        if record.levelno == logging.DEBUG and hasattr(record, 'extra_context'):
            try:
                # Format extra context as JSON
                context = json.dumps(record.extra_context, indent=2)
                record.msg = f"{record.msg}\nContext: {context}"
            except Exception:
                pass
        
        return super().format(record)


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set root logger to DEBUG
    
    formatter = CustomFormatter(
        '%(timestamp)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler for general logs - INFO and above
    file_handler = logging.handlers.RotatingFileHandler(
        "logs/app.log", maxBytes=10485760, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # File handler for error logs - ERROR and above
    error_file_handler = logging.handlers.RotatingFileHandler(
        "logs/error.log", maxBytes=10485760, backupCount=5, encoding="utf-8"
    )
    error_file_handler.setFormatter(formatter)
    error_file_handler.setLevel(logging.ERROR)
    logger.addHandler(error_file_handler)

    # File handler for debug logs - DEBUG and above
    debug_file_handler = logging.handlers.RotatingFileHandler(
        "logs/debug.log", maxBytes=10485760, backupCount=5, encoding="utf-8"
    )
    debug_file_handler.setFormatter(formatter)
    debug_file_handler.setLevel(logging.DEBUG)
    logger.addHandler(debug_file_handler)

    return logger


# Helper function to add context to debug logs
def debug_with_context(logger, message, **context):
    """Log debug message with additional context"""
    extra = {'extra_context': context}
    logger.debug(message, extra=extra)


logger = setup_logging()