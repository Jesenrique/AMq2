"""
predict.py

DESCRIPCIÓN: Este script me permite cargar los datos desde un archivo json, se
utiliza el feature_engineering.py del pipeline de entrenamiento para procesar los
datos cargados. Se carga un modelo serializado previamente entrenado para hacer 
inferencia con los datos que fueron cargados.

AUTOR: Jesús García
FECHA: 12-06-2023
"""

# Imports
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
import log 
from feature_engineering import FeatureEngineeringPipeline

logger=log.logger

class MakePredictionPipeline(object):
    
    def __init__(self, input_path_test,input_path_train,path_json, output_path, 
                 model_path: str = None):
        self.input_path_test = input_path_test
        self.input_path_train = input_path_train
        self.path_json = path_json
        self.output_path = output_path
        self.model_path = model_path
                
                
    def load_data(self)-> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame containing the data from the CSV file.
        Raises:
            IOError: If the CSV file cannot be loaded.
        """
        # Crear el pipeline 
        f_pipeline = FeatureEngineeringPipeline(self.input_path_test, 
                                              self.input_path_train, 
                                              self.output_path)
        # Cargar el dato individual 
        data = pd.read_json(self.path_json, orient='records')
        

        return data
    
    def feauture_engineering(self,data)-> pd.DataFrame:
        """
        feature engineering is applied to the loaded data

        Args:
            data (pandas.DataFrame): receives the load data.
        Returns:
            pandas.DataFrame: the preprocessed data.
        """
        # Crear el pipeline 
        f_pipeline = FeatureEngineeringPipeline(self.input_path_test, 
                                              self.input_path_train, 
                                              self.output_path)
        # Aplicar algunas funciones de feature engineering      
        data=f_pipeline.data_cleaning(data)
        data_processed =f_pipeline.data_transformation(data)
        f_pipeline.write_prepared_data(data)
        
        return data_processed

    def load_model(self) -> None:
        """
        Loads a trained model from a pickle file.
        Raises:
            IOError: If the model file cannot be loaded.
        """
        try: 
            with open(self.model_path, 'rb') as file:  
                self.model = pickle.load(file)
            logger.info("the model.pkl has been loaded successfully")
        except Exception as e:
            logger.error(f"the model.pkl can't be loaded: {e}")
 
        return None


    def make_predictions(self, data_processed):
        """
        Uses a trained model to make predictions on new data.

        Args:
            data (pandas.DataFrame): The DataFrame containing the new data to make predictions on.
        Raises:
            ValueError: If the input DataFrame does not contain the necessary columns.
        """
        try:
            prediction=self.model.predict(data_processed)
            logger.info("prediction made successfully")
        except Exception as e:
            print({e})
            logger.error(f"Error in prediction: {e}")
        
        return prediction


    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Saves the predicted data to a CSV file.

        Args:
            predicted_data (pandas.DataFrame): The DataFrame containing the predicted data.
        Raises:
            IOError: If there is an error writing the file to disk.
        """
        # Guardar el DataFrame en un archivo de Excel en una ubicación específica
        predicted_data=pd.DataFrame(predicted_data)
        try:
            predicted_data.to_csv(self.output_path, index=False)
            logger.info("prediction was saved successfully")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
        return None


    def run(self):
        logger.info("prediction proccess has started")
        raw_data = self.load_data()
        data_processed=self.feauture_engineering(raw_data)
        self.load_model()
        predictions = self.make_predictions(data_processed)
        self.write_predictions(predictions)


if __name__ == "__main__":
    
    pipeline = MakePredictionPipeline(input_path_test = './Test_BigMart.csv',
                                      input_path_train = './Train_BigMart.csv',
                                      path_json='../Notebook/example.json',
                                      output_path = '../src/predictions/predictions.csv',
                                      model_path = '../src/models/model_linearRegression.pkl')
    pipeline.run() 