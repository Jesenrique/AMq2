import pickle
import pandas as pd
path_model="C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP - AMq2\\src\\models\\model.pkl"
path_data="C:\\Users\\Jesus\\Documents\\CEIA\\Aprendizaje de máquina 2\\Material de clase\\TP\\TP - AMq2\\Notebook\\test_final.csv"
data=pd.read_csv(path_data)
data=data.drop("Unnamed: 0",axis=1)
print(data.info())