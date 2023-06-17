# AMq2
Trabajo final de aprendizaje de máquina 2

El trabajo consta de dos pipelines uno de entrenamiento y otro de inferencia.

# pipeline de entrenamiento
(ejecutado en train_pipeline.py)
Este proceso consta de 2 scripts:

Script feauture_engineering.py, realiza el procesamiento de datos realizado por el data scientist, limpiando datos innecesarios, unificando labels, imputando datos daltantes y genera un archivo .csv de todo el procesamiento.

Script train.py, recibe los datos previamente procesados, realiza la división de los datos en entrenamiento y testeo, genera un modelo regresión lineal y guarda el modelo serializado.

(opcional)
Script train_randomF.py, este script realiza el mismo procedimiento train.py, pero en este script se hace uso del framework optuna para optimizar un modelo de random forest regressor con una serie de parametro previamente definidos. Se crea y se guarda un modelo serializado con el cual mas adelante se podran hacer inferencias.

# pipeline de inferencia
(ejecutado en inference_pipeline.py)
Este proceso consta de 1 script:

Script predict.py, se hace el cargue de los datos en este caso un archivo JSON, luego se crea una inferencia de feauture_engineering para poder aplicar todo el procesamiento de los datos. Se carga un modelo entrenado en este caso tenemos la opcion de cargar el modelo de regresión lineal que esta por defecto o se puede cambiar la ruta para hacer inferencia con el modelo de random forest, los modelos disponibles estan en la carpeta ./src/models, una vez hecha la inferencia se guarda en un archivo .csv con ruta ./src/predictions. 

# observaciones
1. Todo el proceso tiene añadido logs que son guardados en el archivo logging_info.log ubicado en ./data/
2. Para ejecutar el pipeline se debe iniciar un cmd desde la carpera data y ejecutar el archivo __init__.py


