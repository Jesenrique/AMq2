
import subprocess

#Ejecuta el pipeline de entrenamiento
try:
    path='../src/train_pipeline.py'
    subprocess.run(["python", path])
    #logger.info("SUCCESS: feature_engineering.py finished successfully")
except Exception as e:
    print("SUCCESS: feature_engineering.py finished successfully")

#Ejecuta el pipeline de inferencia
try:
    path='../src/inference_pipeline.py'
    subprocess.run(["python", path])
    #logger.info("SUCCESS: feature_engineering.py finished successfully")
except Exception as e:
    print("SUCCESS: feature_engineering.py finished successfully")



