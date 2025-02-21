# filepath: logging_config.py
import logging
import logging.handlers
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any
from configuration import LOG_SETTINGS

class CustomFormatter(logging.Formatter):
    """Custom formatter that includes timezone-aware UTC timestamp and formats debug context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with UTC timestamp and optional context."""
        record.timestamp = datetime.now(timezone.utc).isoformat()
        if record.levelno == logging.DEBUG and hasattr(record, 'extra_context'):
            try:
                context = json.dumps(record.extra_context, indent=2, ensure_ascii=False)
                record.msg = f"{record.msg}\nContext: {context}"
            except (TypeError, ValueError) as e:
                record.msg = f"{record.msg}\nContext Error: {str(e)}"
            except Exception:
                record.msg = (f"{record.msg}\nContext: (Could not serialize to"
                               f"JSON: Unknown Exception)")
        return super().format(record)

def create_file_handler(
    log_path: Path,
    level: int,
    formatter: logging.Formatter,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None
) -> logging.Handler:
    """Create a file handler with optional rotation settings."""
    if max_bytes and backup_count:
        handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
    else:
        handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler

def setup_logging(
    log_dir: str = LOG_SETTINGS['DIR'],
    logger_name: Optional[str] = None,
    max_bytes: int = LOG_SETTINGS['MAX_BYTES'],
    backup_count: int = LOG_SETTINGS['BACKUP_COUNT'],
    log_format: str = LOG_SETTINGS['FORMAT'],
    clear_handlers: bool = True
) -> logging.Logger:
    """Set up logging configuration with file and console handlers."""
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(getattr(logging, LOG_SETTINGS['LEVEL']))

    if clear_handlers:
        for handler in logger.handlers[:]:
            handler.close()  # Close handlers to free resources
            logger.removeHandler(handler)

    formatter = CustomFormatter(log_format)
    log_dir_path = Path(log_dir)
    try:
        log_dir_path.mkdir(exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create log directory {log_dir}: {e}")

    handlers: List[Tuple[Path, int, Optional[int], Optional[int]]] = [
        (Path(LOG_SETTINGS['FILE_PATH']), logging.INFO, max_bytes, backup_count),
        (log_dir_path / "error.log", logging.ERROR, max_bytes, backup_count),
        (log_dir_path / "debug.log", logging.DEBUG, max_bytes, backup_count)
    ]

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    for log_path, level, max_size, backups in handlers:
        try:
            handler = create_file_handler(log_path, level, formatter, max_size, backups)
            logger.addHandler(handler)
        except Exception as e:
            logger.error(f"Failed to create handler for {log_path}: {e}")

    return logger

def debug_with_context(logger: logging.Logger, message: str, **context: Any) -> None:
    """Log a debug message with additional context."""
    extra = {'extra_context': context}
    logger.debug(message, extra=extra)

# initialize the root logger.  Individual modules should get their own loggers.
logger = setup_logging()