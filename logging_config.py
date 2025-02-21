# filepath: logging_config.py
import logging
import logging.handlers
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any
from uuid import uuid4
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

# Configure conversation logger
conversation_logger = logging.getLogger('conversation_logger')
conversation_logger.setLevel(logging.INFO)

# Configure rotating file handler for conversations.log
conversation_handler = logging.handlers.RotatingFileHandler(
    Path(LOG_SETTINGS['DIR']) / 'conversations.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
conversation_handler.setFormatter(logging.Formatter('%(message)s'))
conversation_logger.addHandler(conversation_handler)

def generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return str(uuid4())

def parse_streaming_response(response_chunks: List[str]) -> str:
    """
    Parse streaming response chunks and extract the actual content.
    
    Args:
        response_chunks: List of response chunks in streaming format
    Returns:
        Extracted content as a single string
    """
    content = []
    for chunk in response_chunks:
        if chunk.startswith('data: ') and not chunk.strip().endswith('[DONE]'):
            try:
                # Parse the JSON data after 'data: '
                data = json.loads(chunk[6:].strip())
                if 'delta' in data and 'content' in data['delta']:
                    content.append(data['delta']['content'])
            except json.JSONDecodeError:
                continue
    return ''.join(content)

def log_conversation_entry(conversation_id: str, user_prompt: str, ai_response: str) -> None:
    """
    Log a conversation entry to conversations.log in JSON format.
    
    Args:
        conversation_id: Unique identifier for the conversation
        user_prompt: The user's input message
        ai_response: The complete AI response
    """
    try:
        # Parse the streaming response to extract clean content
        clean_response = parse_streaming_response(ai_response.split('\n'))
        
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'conversation_id': conversation_id,
            'user_prompt': user_prompt,
            'ai_response': clean_response
        }
        conversation_logger.info(json.dumps(entry, ensure_ascii=False))
    except Exception as e:
        # Log any errors to the main application log
        logging.getLogger().error(f"Failed to log conversation entry: {str(e)}", 
                                extra={'conversation_id': conversation_id})

# initialize the root logger.  Individual modules should get their own loggers.
logger = setup_logging()