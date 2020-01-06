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
        # pd.DataFrame(transformed_df,columns=temp_col_names).to_csv(transformation_type+'_out.csv')
    return pd.concat(arr_transformed_df,axis=1)

    

def preprocess(df,obj_imputer,obj_scaler,):
    df_model_data = df[variable_list]
    # print(df_model_data.head())
    for col in df_model_data.columns:
        df_model_data[col] = pd.to_numeric(df_model_data[col],errors='ignore')
    # print(type(df_model_data))
    imputed_df_model_data = pd.DataFrame(obj_imputer.transform(df_model_data),columns=variable_list)
    # imputed_df_model_data.to_csv('Imputed_out.csv')
    # print(type(imputed_df_model_data))
    scaled_df_model_data = pd.DataFrame(obj_scaler.transform(imputed_df_model_data),columns=variable_list)
    # scaled_df_model_data.to_csv('Scaled_out.csv')
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
    bins = [-0.001,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.105,0.11,0.115,0.12,0.125,0.13,0.135,0.14,0.145,0.15,0.155,0.16,0.165,0.17,0.175,0.18,0.185,0.19,0.195,0.2,0.205,0.21,0.215,0.22,0.225,0.23,0.235,0.24,0.245,0.25,0.255,0.26,0.265,0.27,0.275,0.28,0.29,0.295,0.3,0.31,0.32,0.33,0.34,0.35,100]
    labels = ['0.0595','0.0627','0.0676','0.0712','0.0775','0.0816','0.0879','0.0943','0.1005','0.1053','0.1102','0.1155','0.1212','0.126','0.1301','0.1363','0.1415','0.1466','0.15','0.1568','0.16085','0.1649','0.171','0.1788','0.1869','0.1968','0.2009','0.2114','0.2217','0.2317','0.2394','0.2426','0.2482','0.2545','0.2616','0.2679','0.2787','0.288','0.29225','0.2958','0.2991','0.3047','0.3078','0.3082','0.3091','0.316','0.31735','0.3178','0.3189','0.31915','0.3203','0.3249','0.3488','0.6789']
    predictions = [i/100.0 for i in predictions]
    # df_out = pd.DataFrame(predictions)
    # pd.cut(df_out[0],bins=bins,labels=labels,duplicates='drop')
    # print(df_out[0].head())
    # print(pd.cut(pd.DataFrame(predictions)[0],bins=bins,labels=labels,precision=5).head())
    actual_50_output = pd.cut(pd.DataFrame(predictions)[0],bins=bins,labels=labels,precision=5,duplicates='drop')
    str_predictions = [str(i) for i in actual_50_output.values]
    return dict(zip(range(50),str_predictions))
    # return jsonify(dict(zip(range(50),predict(df))))

# print(len(df.columns))
# print(predict(df))

if __name__ == '__main__':
    app.run(debug=True)
