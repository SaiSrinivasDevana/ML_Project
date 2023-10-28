import logging
import os
from datetime import datetime

logfile=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",logfile)
os.makedirs(logs_path,exist_ok=True)

LOGS_File_Path=os.path.join(logs_path,logfile)

logging.basicConfig(
    filename=LOGS_File_Path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s- %(message)s",
    level=logging.INFO,
)