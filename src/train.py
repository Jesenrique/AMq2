"""
train.py

DESCRIPCIÓN: El script recibe los datos previamente procesados, realiza la división
de los datos en entrenamiento y testeo, genera un modelo regresión lineal y 
guarda el modelo serializado.

AUTOR: Jesús García.
FECHA: 12-06-2023
"""

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import pickle
import log

logger= log.logger

class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path
        
        

    def read_data(self) -> pd.DataFrame:
        """
        Reads data from a CSV file and returns a pandas DataFrame object.

        Returns:
            pandas.DataFrame: The DataFrame object that contains the data from the CSV file.
        Raises:
            Exception: If any error occurs while reading the CSV file.
        """
        try:
            pandas_df=pd.read_csv(self.input_path)
            logger.info("SUCCESS: data was loaded successfully")
        except Exception as e:
            logger.error(f"The data couldn't be loaded:{e}")

        return pandas_df

    
    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trains a linear regression model using the provided DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data to train the model.
        Returns:
            model: the trained model.
        """
        # División del dataset de train y test
        df_train = df.loc[df['Set'] == 'train']
        df_test = df.loc[df['Set'] == 'test']

        # Eliminando columnas sin datos
        df_train.drop(['Set'], axis=1, inplace=True)
        df_test.drop(['Item_Outlet_Sales','Set'], axis=1, inplace=True)

        # Guardando los datasets
        #df_train.to_csv("train_final.csv")
        #df_test.to_csv("test_final.csv")

        #Instanciamiento del modelo
        seed = 28
        model = LinearRegression()

        # División de dataset de entrenaimento y validación
        X = df_train.drop(columns='Item_Outlet_Sales') #[['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = train_test_split(X, df_train['Item_Outlet_Sales'], test_size = 0.3, random_state=seed)
        logger.info("The data was divided in test and train sets")

        # Entrenamiento del modelo
        model.fit(x_train,y_train)
        logger.info("The model was trained")

        # Predicción del modelo ajustado para el conjunto de validación
        pred = model.predict(x_val)

        # Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        R2_train = model.score(x_train, y_train)
        #print('Métricas del Modelo:')
        #print('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))
        logger.info(f"The train metrics of the model are: RMSE: {mse_train**0.5}"
                     f"-R2:{R2_train}")

        mse_val = metrics.mean_squared_error(y_val, pred)
        R2_val = model.score(x_val, y_val)
        #print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))
        logger.info(f"The test metrics of the model are: RMSE: {mse_val**0.5}"
                     f"-R2:{R2_val}")

        return model

    def model_dump(self, model_trained) -> None:
        """
        Saves the trained model to disk using pickle.
        
        Args:
            model_trained: The trained model to be saved.
        Raises:
            IOError: If there is an error writing the file to disk.
        """
        try:
            with open(self.model_path, 'wb') as file:
                pickle.dump(model_trained, file)
            logger.info("SUCCESS: The model has been saved successfully")
        except Exception as e:
            logger.error(f"An error occurred while saving the model:{e}")
   
        return None

    def run(self):
        logger.info("SUCCESS: The training of the linear regression model has started")
        df = self.read_data()
        model_trained= self.model_training(df)
        self.model_dump(model_trained)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = './preprocessed_data.csv',
                          model_path = '../src/models/model_linearRegression.pkl').run()