import pickle
import csv
from settings import *
import pandas as pd
import json
from flask import Flask,jsonify
from flask import request

app = Flask(__name__)
variable_list = None

pca_group = ['Grp_Unsecured', 'Grp_Tradelines', 'Grp_Utilization', 'Grp_Dqs',
       'Grp_Collections', 'Grp_Inqs', 'Grp_PublicRecords']
dict_pca_model = {}

def init_objects():
   
    obj_scaler = pickle.load(open(scaler_objfile_path,'rb'))
    
    obj_grp_collections_pca  = pickle.load(open(grp_collections_pca_file_path,'rb'))
    obj_grp_dqs_pca = pickle.load(open(grp_dqs_pca_file_path,'rb'))
    obj_grp_inqs_pca = pickle.load(open(grp_inqs_pca_file_path,'rb'))
    obj_grp_public_records_pca = pickle.load(open(grp_public_records_pca_file_path,'rb'))
    obj_grp_tradelines_pca = pickle.load(open(grp_tradelines_pca_file_path,'rb'))
    obj_grp_unsecured_pca = pickle.load(open(grp_unsecured_pca_file_path,'rb'))
    obj_grp_utilization_pca = pickle.load(open(grp_utilization_pca_file_path,'rb'))
    obj_imputer  = pickle.load(open(imputer_objfile_path,'rb'))
    obj_imputer.set_params(add_indicator=None)
    # print(obj_imputer)
    xgb_model  = pickle.load(open(xgb_objfile_path,'rb'))
    df_pca_variables = pd.read_csv('VariableGroups_PCA.csv')
    df_pca_variables['Variable List'] = df_pca_variables['Variable List'].str.strip()
    for p in pca_group:
        temp_dict = {}
        temp_dict['variables'] = list(df_pca_variables[df_pca_variables['Groups'] == p]['Variable List'])
        if p == 'Grp_Unsecured':
            temp_dict['model'] = obj_grp_unsecured_pca
        elif p == 'Grp_Tradelines':
            temp_dict['model'] = obj_grp_tradelines_pca
        elif p == 'Grp_Utilization':
            temp_dict['model'] = obj_grp_utilization_pca
        elif p == 'Grp_Dqs':
            temp_dict['model'] = obj_grp_dqs_pca
        elif p == 'Grp_Collections':
            temp_dict['model'] = obj_grp_collections_pca
        elif p == 'Grp_Inqs':
            temp_dict['model'] = obj_grp_inqs_pca
        else:
            temp_dict['model'] = obj_grp_public_records_pca
        dict_pca_model[p] = temp_dict
    return obj_imputer,obj_scaler,xgb_model


with open('Variable_List.csv', 'r') as f:
    reader = csv.reader(f)
    variable_list = list(reader)

variable_list = variable_list[0]
for ix,val in enumerate(variable_list):
    variable_list[ix] = val.strip()

def get_pca_transformation(df_scaled_data):
    arr_transformed_df = []
    for transformation_type in pca_group:
        lst_variables = dict_pca_model[transformation_type]['variables']
        # print(type(lst_variables))
        # print(df_scaled_data.head())
        df_temp = df_scaled_data[lst_variables]
        transformed_df = dict_pca_model[transformation_type]['model'].transform(df_temp)
        temp_col_names = []
        for i in range(dict_pca_model[transformation_type]['model'].n_components_):
            temp_col_names.append(transformation_type + '_' + str(i))

        arr_transformed_df.append(pd.DataFrame(transformed_df,columns=temp_col_names))
    return pd.concat(arr_transformed_df,axis=1)

    

def preprocess(df,obj_imputer,obj_scaler,):
    df_model_data = df[variable_list]
    # print(df_model_data.head())
    for col in df_model_data.columns:
        df_model_data[col] = pd.to_numeric(df_model_data[col],errors='ignore')
    # print(type(df_model_data))
    imputed_df_model_data = pd.DataFrame(obj_imputer.transform(df_model_data),columns=variable_list)
    # print(type(imputed_df_model_data))
    scaled_df_model_data = pd.DataFrame(obj_scaler.transform(imputed_df_model_data),columns=variable_list)
    # component_df_model_data = obj_pca.transform(scaled_df_model_data)
    return scaled_df_model_data,get_pca_transformation(scaled_df_model_data)

def predict(x):
    scaled_data,data = preprocess(x,obj_imputer,obj_scaler)
    # print(data.head())
    final_data = pd.concat([scaled_data[['Vantage','ActiveCreditLimit']],data],axis=1)
    return xgb_model.predict(final_data)

# print('Initializing Objects')
obj_imputer,obj_scaler,xgb_model = init_objects()
# print('########################')
# print(obj_imputer)
@app.route('/api/v1.0/predictions', methods=['POST'])
def api_preidct():
    # print(request.json)
    model_query_data = request.json
    df = pd.DataFrame(model_query_data)
    # print(df.head())
    # return jsonify({'status':'OK'})
    # urn jsonify(pd.DataFrame(predict(df))
    predictions = predict(df)
    str_predictions = [str(i) for i in predictions]
    return dict(zip(range(50),str_predictions))
    # return jsonify(dict(zip(range(50),predict(df))))

# print(len(df.columns))
# print(predict(df))

if __name__ == '__main__':
    app.run(debug=True)
