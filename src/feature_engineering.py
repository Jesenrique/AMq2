"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR: Jesús García 
FECHA: 12-06-2023
"""

# Imports
import pandas as pd
import csv
import log

logger = log.logger

class FeatureEngineeringPipeline(object):

    def __init__(self, input_path_test,input_path_train, output_path):
        """
        Initialize the FeatureEngineeringPipeline class with the input and output file paths.

        Args:
            input_path_test: Path of the test data CSV file.
            input_path_train: Path of the training data CSV file.
            output_path: Path of the output CSV file for the preprocessed data."
        """
        self.input_path_test = input_path_test
        self.input_path_train = input_path_train
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Read the training and test data from the CSV files.

        Returns:
            pd.DataFrame: The desired data as a pandas DataFrame.
        """
        try:
            data_train = pd.read_csv(self.input_path_train)
            logger.info("SUCCESS: data train was loaded successfully")
        except Exception as e:
            logger.error(f"The data train couldn't be loaded:{e}")  

        try:
            data_test = pd.read_csv(self.input_path_test)
            logger.info("SUCCESS: data test was loaded successfully")
        except Exception as e:
            logger.error(f"The data test couldn't be loaded:{e}")   
            
        # Identificando la data de train y de test, para posteriormente unión y separación
        data_train['Set'] = 'train'
        data_test['Set'] = 'test'   

        df = pd.concat([data_train, data_test], ignore_index=True, sort=False)

        return df

    def data_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform imputation of missing values in the data.

        Args:
            data (pd.DataFrame): Raw dataset.
        Returns:
            pd.DataFrame: Dataset with imputed missing data.
        """
       
        #Imputación de datos por moda en la feauture item_weight
        productos = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for producto in productos:
            moda = (data[data['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
            data.loc[data['Item_Identifier'] == producto, 'Item_Weight'] = moda
        
        #Imputación de datos en la feature outlet_identifier 
        outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        for outlet in outlets:
            data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        return data

    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data cleaning on the DataFrame.

        Args:
            df (pd.DataFrame): Dataset without missing data.
        Returns:
            pd.DataFrame: Dataset with The number of years of the store and tags
            for item_content_fat are unified
        """
        #Numero de años a la fecha
        df['Outlet_Establishment_Year'] = 2020 - df['Outlet_Establishment_Year']

        #Unificación de etiquetas Fat_content
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

        return df

    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform variable transformation on the data.
        
        Args:
            data (pd.DataFrame): Dataset after the cleaning process.
        Returns:
            pd.DataFrame: The number of years of the store is obtained and tags 
            for item_content_fat are unified.
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

        # Codificación de los precios 
        data['Item_MRP'], bins = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4],retbins=True)

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
        Saves the prepared DataFrame to a CSV file at a specific location.

        Args:
            transformed_dataframe (pd.DataFrame): The DataFrame containing the prepared data.
        Returns:
            None
        Raises:
            Exception: If the preprocessed data couldn't be saved to the specified location.
        """
        try:
            transformed_dataframe.to_csv(self.output_path, index=False)
            logger.info("SUCCESS: preprocessed data was saved successfully")
        except Exception as e:
            logger.error(f"SUCCESS: preprocessed data couldn't be loaded.{e}") 

        return None

    def run(self):
        logger.info("SUCCESS: feauture engineering has started")
        df = self.read_data()
        data_clean = self.data_cleaning(df)
        logger.info("SUCCESS: data was cleaning")
        data = self.data_imputation(data_clean)
        logger.info("SUCCESS: data missing was restored")
        df_transformed = self.data_transformation(data)
        logger.info("SUCCESS: data was transformed")
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path_test = './Test_BigMart.csv',
                               input_path_train = './Train_BigMart.csv',
                               output_path = './preprocessed_data.csv').run()