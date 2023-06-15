"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
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
        COMPLETAR DOCSTRING
        """
        try:
            data=pd.read_csv(self.input_path)
            logger.info("the data was loaded successfully")
        except Exception as e:
            logger.error(f"the data can't be loaded: {e}")

        data=data.drop("Unnamed: 0",axis=1)
        #print(data.info())
        return data

    def load_model(self) -> None:
        """
        COMPLETAR DOCSTRING
        """
        # Carga el modelo desde el .pkl
        try: 
            with open(self.model_path, 'rb') as file:  
                self.model = pickle.load(file)
            logger.info("the model.pkl has been loaded successfully")
        except Exception as e:
            logger.error(f"the model.pkl can't be loaded: {e}")
 
        return None


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """
        try:
            new_data=self.model.predict(data)
            logger.info("prediction made successfully")
        except Exception as e:
            logger.error(f"Error in prediction: {e}")

        return new_data


    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
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
    
    pipeline = MakePredictionPipeline(input_path = 'C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP-AMq2\\Notebook\\test_final.csv',
                                      output_path = 'C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP-AMq2\\src\\predictions\predictions.csv',
                                      model_path = 'C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP-AMq2\\src\\models\\model.pkl')
    pipeline.run()  
