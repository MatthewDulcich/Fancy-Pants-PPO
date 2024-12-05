import logging
import os
import time

def configure_logging():
    # Ensure the logs directory exists
    logs_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Generate a unique log file name
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(logs_dir, f"fpa_game_logs_{current_time}.log")

    # Clear existing handlers if reinitializing
    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers.clear()

    # Configure logging to write only to the file
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),  # Logs written only to the file
        ],
        level=logging.INFO  # Default log level
    )

    return logger, log_filename  # Return the logger and log file name