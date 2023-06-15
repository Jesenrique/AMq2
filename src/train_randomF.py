"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import pickle
import optuna
from sklearn.ensemble import RandomForestRegressor
import log

logger = log.logger

class ModelTrainingPipeline(object):

    def __init__(self, input_path,model_path):
        self.input_path = input_path
        self.model_path = model_path
 
    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
        # COMPLETAR CON CÓDIGO
        try:
            pandas_df=pd.read_csv(self.input_path)
            logger.info("SUCCESS: data was loaded successfully")
        except Exception as e:
            logger.error(f"The data couldn't be loaded:{e}")

        return pandas_df

    
    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        # Eliminación de variables que no contribuyen a la predicción por ser muy específicas
        dataset = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        # División del dataset de train y test
        df_train = dataset.loc[dataset['Set'] == 'train']
        df_test = dataset.loc[dataset['Set'] == 'test']

        # Eliminando columnas sin datos
        df_train.drop(['Set'], axis=1, inplace=True)
        df_test.drop(['Item_Outlet_Sales','Set'], axis=1, inplace=True)

        # Guardando los datasets
        df_train.to_csv("train_final.csv")
        df_test.to_csv("test_final.csv")

        # División de dataset de entrenaimento y validación
        X = df_train.drop(columns='Item_Outlet_Sales') #[['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(X, df_train['Item_Outlet_Sales'], test_size = 0.3, random_state=11)
        logger.info("The data was divided in test and train sets")
        
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        logger.info("Model optimization has started")
        study.optimize(self.objective, n_trials=10)
        # Imprimir los resultados de la optimización
        print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
        logger.info(f"The best parametres are: {study.best_params}")

        model = RandomForestRegressor(**study.best_params)
        logger.info("The model was created successfully")
        model.fit(self.x_train, self.y_train)

        return model

    def objective(self,trial):
        """
        Objective function for optimizing hyperparameters of a RandomForestClassifier using Optuna.
        
        Args:
            trial: A `Trial` object from Optuna that contains the state of the optimization trial.
        
        Returns:
            The accuracy score of the RandomForestClassifier using the hyperparameters suggested by Optuna.
        """
        # Definir los hiperparámetros a optimizar
        n_estimators = trial.suggest_int('n_estimators', 5, 500)
        max_depth = trial.suggest_int('max_depth', 2, 500)
        min_samples_split = trial.suggest_float("min_samples_split", 0.01, 1)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        criterion = trial.suggest_categorical("criterion", ['poisson','squared_error'])
        
        # Crear el clasificador con los hiperparámetros sugeridos por Optuna
        clf = RandomForestRegressor(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        criterion=criterion
                                        )
    
        # Entrenar el clasificador y calcular la precisión en el conjunto de prueba
        clf.fit(self.x_train, self.y_train)
        score = cross_val_score(clf, self.x_train, self.y_train, cv=3)
        accuracy = score.mean()

        return accuracy

    def model_dump(self, model_trained) -> None:
        """
        COMPLETAR DOCSTRING
        """
        try:
            with open(self.model_path, 'wb') as file:
                pickle.dump(model_trained, file)
            logger.info("The model has been saved successfully")
        except Exception as e:
            logger.error(f"An error occurred while saving the model:{e}")
        return None
     
    def run(self):
        logger.info("The training of the Random Forest model has started")
        df = self.read_data()
        model=self.model_training(df)
        self.model_dump(model)


if __name__ == "__main__":
    ModelTrainingPipeline(input_path = 'C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP-AMq2\\data\\preprocessed_data.csv',
                          model_path = 'C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP-AMq2\\src\\models\\model_RandomForest.pkl').run()