import boto3
import pandas
from sentence_transformers import SentenceTransformer
from io import StringIO
import numpy 
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import sagemaker
import json
import code_variables

def input_fn(request_body, request_content_type):
   # """An input_fn that loads a string json array"""
    print("calling input function")
    if request_content_type == "application/string":
        string_data= request_body.decode()
        return string_data
    elif request_content_type == "application/json":
        content = json.loads(request_body)
        return (content['material_description'])
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        return "Input data format should be in string or json"


def predict_fn(input_object, model): 

    ########################################### 

    # Do your custom preprocessing logic here # 

    ########################################### 
    # print(f'Input is {input_object}')
    # fmcc_bucket='aiops.d-aiml-315952988300-us-east-2'
    # path='data/comm_embeds.csv'
    # s3_client=boto3.client('s3')
    output={}
    try:
        suggestion_df=model.comm_code(input_object)
        output['suggestions']=[]
        top_rows=suggestion_df.head(3)
        for _,row in top_rows.iterrows():
            output['suggestions'].append({
                'Material Description': row['material_description'],
                'Match Confidence': row['similarity'],
                'Supplier Part No': row['supplier_part_number'],
                'Supplier ID': row['vendor_number'],
                'Supplier Name': row['supplier_name_l1'],
                'Requisition Date': row['requisition_date']
            })
        return(output)
    except Exception as e:
        print(f'{e} encountered')
        # print('Main function error')

def model_fn(model_dir):
    print("loading model from: {}".format(model_dir))
    # fmcc_bucket='aiops.d-aiml-315952988300-us-east-2'
    # path='data/comm_embeds.csv'
    sagemaker_session=sagemaker.session.Session()
    loaded_model = joblib.load(os.path.join(model_dir, f"{code_variables.model_name}.pkl"))
    print("model: ", loaded_model)
    loaded_model.initial_load(bucket=code_variables.bucket_name,file_path=code_variables.save_location,
                              sagemaker_session=sagemaker_session,file_name=code_variables.embed_file)
    return loaded_model
    