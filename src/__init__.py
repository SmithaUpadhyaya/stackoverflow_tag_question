from pathlib import Path
import logging
import sys


logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"

i = 2 #Since project structure are max at 2 level
cur_path = Path().resolve().parent

while i > 0: 

    if Path.exists(cur_path / 'src'):
        break
    
    cur_path = cur_path.parent
    i = i - 1

#log_filepath = Path().resolve().parent.parent.joinpath(log_dir) #Since this file is 2 level bellow the parent folder
log_filepath = cur_path.joinpath(log_dir)

log_filepath.mkdir(parents = True, exist_ok = True)

log_filepath = log_filepath.joinpath("running_logs.log")


logging.basicConfig(
    level = logging.INFO,
    format = logging_str,

    handlers = [
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("projectlogger")
            