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
import log

logger= log.logger

class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
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
        print('Métricas del Modelo:')
        print('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))
        logger.info(f"The train metrics of the model are: RMSE: {mse_train**0.5}"
                     f"-R2:{R2_train}")

        mse_val = metrics.mean_squared_error(y_val, pred)
        R2_val = model.score(x_val, y_val)
        print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))
        logger.info(f"The test metrics of the model are: RMSE: {mse_val**0.5}"
                     f"-R2:{R2_val}")
        #print('\nCoeficientes del Modelo:')
        # Constante del modelo
        #print('Intersección: {:.2f}'.format(model.intercept_))

        # Coeficientes del modelo
        #coef = pd.DataFrame(x_train.columns, columns=['features'])
        #coef['Coeficiente Estimados'] = model.coef_
        #print(coef, '\n')
        #coef.sort_values(by='Coeficiente Estimados').set_index('features').plot(kind='bar', title='Importancia de las variables', figsize=(12, 6))

        return model, x_val, y_val

    def model_dump(self, model_trained,x_vali,y_vali) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        try:
            with open(self.model_path, 'wb') as file:
                pickle.dump(model_trained, file)
            logger.info("SUCCESS: The model has been saved successfully")
        except Exception as e:
            logger.error(f"An error occurred while saving the model:{e}")
        '''
        try: 
            with open(self.model_path, 'rb') as file:  
                model = pickle.load(file)
            print("El modelo serializado ha sido cargado de manera exitosa!")
        except:
            print("Error al momento de cargar el modelo serializado!")

        # Predicción del modelo ajustado para el conjunto de validación
        print(x_vali.info())
        pred=model.predict(x_vali)
        mse_val = metrics.mean_squared_error(y_vali, pred)
        R2_val = model.score(x_vali, y_vali)
        print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))
        '''
        return None

    def run(self):
        logger.info("SUCCESS: The training of the linear regression model has started")
        df = self.read_data()
        model_trained,x_vali, y_vali= self.model_training(df)
        self.model_dump(model_trained,x_vali,y_vali)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = 'C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP-AMq2\\data\\preprocessed_data.csv',
                          model_path = 'C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP-AMq2\\src\\models\\model_linearRegression.pkl').run()