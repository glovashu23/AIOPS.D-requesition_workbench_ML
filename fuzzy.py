#!/usr/bin/env python3.7
import pandas 
import boto3
import numpy 
import sentence_transformers
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
import sagemaker
import os

class fmcc:
    
    def __init_(self):
        self.dataset=''
        self.model=''
    
    #Initialising parameters and also reading the data
    def initial_load(self,bucket,file_path,sagemaker_session,file_name):

        if not os.path.exists('/opt/ml/processing/input'):
            print(f'/opt/ml/processing/input does not exists')
            os.makedirs('/opt/ml/processing/input',exist_ok=True)
            
        print(f"Downloading file from location s3://{bucket}/{file_path}")
        sagemaker_session.download_data('/opt/ml/processing/input', bucket, file_path)
        
        self.dataset=pandas.read_csv(f'/opt/ml/processing/input/{file_name}')
        print(f"Loaded dataset with shape {self.dataset.shape}")
        self.dataset['description_embeds']=self.dataset['embeddings'].apply(self.convert_to_array)
        self.dataset.drop('embeddings',axis=1,inplace=True)
        self.model = sentence_transformers.SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')
    
    #Function to convert embeddings back to a numpy array since after saving and then reading, the type gets converted to string
    def convert_to_array(self,embeddings):

        embeddings=embeddings.replace('[','')
        embeddings=embeddings.replace(']','')
        embeddings=embeddings.replace(',','')
        str_embed=embeddings.split()
        return(numpy.array(str_embed,dtype=float))
    
    def comm_code(self,in_desc):

        output={}
        output['suggestions']=[]
        set_of_embeds=[self.model.encode(in_desc)]
        set_of_embeds.extend(self.dataset['description_embeds'].tolist())
        # set_of_embeds.extend([self.convert_to_array(x) for x in self.dataset['embeddings'].tolist()])
        cosine_values=cosine_similarity([set_of_embeds[0]],set_of_embeds[1:])
        print(f"Getting top 3 matches for {in_desc}")
        self.dataset['similarity']=cosine_values[0].tolist()
        self.dataset.sort_values(by='similarity',ascending=False,inplace=True)
        output=self.dataset.head(3)
        return output
            