"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
import log

logger=log.logger

class FeatureEngineeringPipeline(object):

    def __init__(self, path_json, output_path):
        self.output_path = output_path
        self.path_json = path_json

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
        # leyendo los datos del archivo JSON
        try:
            df = pd.read_json(self.path_json, orient='records')
            logger.info("SUCCESS: data train was loaded successfully")
        except Exception as e:
            logger.error(f"The data train couldn't be loaded:{e}")   

        return df

    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        #Numero de años a la fecha
        df['Outlet_Establishment_Year'] = 2020 - df['Outlet_Establishment_Year']

        #Unificación de etiquetas Fat_content
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

        return df

    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        # Asignación de nueva categoria en Item_type igual NA
        data.loc[data['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'
        

        # FEATURES ENGINEERING: creando categorías para 'Item_Type'
        data['Item_Type'] = data['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
        'Seafood': 'Meats', 'Meat': 'Meats',
        'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
        'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
        'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

        # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        data.loc[data['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        # Codificación de los precios segun los rangos en el analisis de datos
        def item_value (x):
            if x < 31.29:
                x = 1
            elif  31.29 <= x <94.012:
                x = 2
            elif  94.012 <= x < 185.856:
                x = 3
            else:
                x = 4
            return 
        
        data['Item_MRP']=data['Item_MRP'].apply(item_value)

        # Codificacción de variables ordinales y numericas
        ## Se crea copia del dataframe para valores codificados
        dataframe = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()

        ## Codificación de variables ordinales
        dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos

        ## Codificación de variables nominales
        dataframe = pd.get_dummies(dataframe, columns=['Outlet_Type'])

        return dataframe

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        # Guardar el DataFrame en un archivo de Excel en una ubicación específica
        try:
            transformed_dataframe.to_csv(self.output_path, index=False)
            logger.info("SUCCESS: preprocessed data was saved successfully")
        except Exception as e:
            logger.error(f"preprocessed data couldn't be loaded.{e}") 

        return None

    def run(self):
        logger.info("SUCCESS: feauture engineering inference has started")
        df = self.read_data()
        data_clean = self.data_cleaning(df)
        logger.info("SUCCESS: data was cleaning")
        df_transformed = self.data_transformation(data_clean)
        logger.info("SUCCESS: data missing was restored")
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(path_json="C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP-AMq2\\Notebook\\example.json",
                                output_path = 'C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP-AMq2\\data\\preprocessed_data_2.csv').run()