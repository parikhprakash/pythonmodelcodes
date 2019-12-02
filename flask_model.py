import pickle
import csv
from settings import *
import pandas as pd
import json

obj_scaler = None
obj_imputer = None
obj_pca = None
xgb_model = None



def init_objects():
    obj_scaler = pickle.load(open(scaler_objfile_path,'rb'))
    obj_pca  = pickle.load(open(pca_objfile_path,'rb'))
    obj_imputer  = pickle.load(open(imputer_objfile_path,'rb'))
    obj_imputer.set_params(add_indicator=None)
    xgb_model  = pickle.load(open(xgb_objfile_path,'rb'))
    with open('Variable_List.csv', 'r') as f:
        reader = csv.reader(f)
        variable_list = list(reader)

variable_list = variable_list[0]
for ix,val in enumerate(variable_list):
    variable_list[ix] = val.strip()

def preprocess(df):
    df_model_data = df[variable_list]
    for col in df_model_data.columns:
        df_model_data[col] = pd.to_numeric(df_model_data[col],errors='ignore')
    imputed_df_model_data = obj_imputer.transform(df_model_data)
    scaled_df_model_data = obj_scaler.transform(imputed_df_model_data)
    component_df_model_data = obj_pca.transform(scaled_df_model_data)
    return component_df_model_data

def predict(x):
    data = preprocess(x)
    return xgb_model.predict(data)

init_objects()
model_query_data = json.load(open('Example2_Formatted.json','r'))
df = pd.DataFrame(model_query_data)
print(predict(df))
