import subprocess
import log

logger=log.logger

try:
    subprocess.run(['python','../src/feature_engineering.py'])
    logger.info("SUCCESS: feature_engineering.py finished successfully")
except Exception as e:
    logger.error(f"the script feature_engineering.py has an error: {e}")

try:
    subprocess.run(['Python', '../src/train.py'])
    logger.info("SUCCESS: train.py finished successfully")
except Exception as e:
    logger.error(f"the script train.py has an error: {e}")
