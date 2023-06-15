import logging

logging.basicConfig(
    filename='logging_info.log',
    level=logging.INFO, 
    filemode='a',   
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',     
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)