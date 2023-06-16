"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR: Jesús García
FECHA: 12-06-2023
"""

# Imports
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
import log 

logger=log.logger

class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
                
                
    def load_data(self):
        """
        Loads a CSV file into a pandas DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame containing the data from the CSV file.
        Raises:
            IOError: If the CSV file cannot be loaded.
        """
        try:
            data=pd.read_csv(self.input_path)
            logger.info("the data was loaded successfully")
        except Exception as e:
            logger.error(f"the data can't be loaded: {e}")

        #data=data.drop("Unnamed: 0",axis=1)
        #print(data.info())
        return data

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


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Uses a trained model to make predictions on new data.

        Args:
            data (pandas.DataFrame): The DataFrame containing the new data to make predictions on.
        Returns:
            pandas.DataFrame: The DataFrame containing the predictions.
        Raises:
            ValueError: If the input DataFrame does not contain the necessary columns.
        """
        try:
            new_data=self.model.predict(data)
            #logger.info("prediction made successfully")
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
        
        return new_data


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
        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    
    #spark = Spark()
    
    pipeline = MakePredictionPipeline(input_path = '../Notebook/test_final.csv',
                                      output_path = '../src/predictions/predictions.csv',
                                      model_path = '../src/models/model.pkl')
    pipeline.run()  
