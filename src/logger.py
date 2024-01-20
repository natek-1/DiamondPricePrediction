import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
path = os.path.join(os.getcwd(), "logs")
os.makedirs(path, exist_ok=True)

LOG_FILE_PATH = os.path.join(path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level = logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)

