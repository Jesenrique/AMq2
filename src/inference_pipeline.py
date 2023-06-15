import subprocess
import log

logger=log.logger

try:
    subprocess.run(['python','../src/feature_engineering_2.py'])
    logger.info("SUCCESS: feature_engineering.py finished successfully")
except Exception as e:
    logger.error(f"the script feature_engineering.py has an error: {e}")

try:
    subprocess.run(['python','../src/predict.py'])
    logger.info("SUCCESS: predict.py finished successfully")
except Exception as e:
    logger.error(f"the script predict.py has an error: {e}")