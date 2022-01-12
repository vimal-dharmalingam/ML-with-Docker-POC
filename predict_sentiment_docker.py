#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import s3fs
import numpy as np
import pandas as pd
import re
import boto3
import pickle 
from datetime import timedelta, datetime
import os
import sys
from io import StringIO
from io import BytesIO
import time
from botocore.exceptions import ClientError

# text cleansing libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# downloading required word clouds
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


var_dict={

    "result_file_location" : "ml/analytical-result-store",
    "result_file_name"    : "amazon_sentiment_analysis_prediction_pynb_result.csv",
    "pretrained_model_loc" : "ml/prediction_model/amazon_sentiment_analysis_prediction_model.pkl",
    "inference_data_folder":"ml/prediction-data",
    "inference_file_name":"test_data_sentiment_analysis.csv",
    "s3_bucket_name" : "swire-datalake-dev-bucket",
     "inference_file_backup_path" : "ml/prediction-data-backup",
    "inference_file_backup_key" : "test_data_sentiment_analysis.csv"
    #"aws_id": "AKIA4EEJ3XXWKQ37XN2J",
    #"aws_secret_key" :"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"
    }


def read_csv_file(bucket_name,inference_data_key): #,aws_access_id,aws_access_key
    """ Read the csv predcition file from s3 using boto3 client
    """
    client = boto3.client('s3') #, aws_access_key_id = aws_access_id , aws_secret_access_key= aws_access_key

    csv_obj = client.get_object(Bucket=bucket_name, Key=inference_data_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')

    Raw_data = pd.read_csv(StringIO(csv_string))
    
    return Raw_data


def load_pickle_data(bucket_name, model_key): # ,aws_access_id,aws_access_key
    """
    Get the stored pretrained model from S3 bucket
    """
    client = boto3.client('s3') #, aws_access_key_id = aws_access_id, aws_secret_access_key=aws_access_key
    response = client.get_object(Bucket=bucket_name, Key=model_key)
    body = response['Body'].read()
    trained_model = pickle.loads(body)

    return trained_model


def data_cleansing(Raw_data,trained_tfidf_vector): 
    
    test_data=Raw_data.drop('Unnamed: 0',axis=1)
    #lower case all text
    test_data["reviews.text"]=test_data["reviews.text"].str.lower() 

    #tokenization of words
    test_data['reviews.text'] = test_data.apply(lambda row: word_tokenize(row['reviews.text']), axis=1) 

    #only alphanumerical values
    test_data["reviews.text"] = test_data['reviews.text'].apply(lambda x: [item for item in x if item.isalpha()]) 

    #lemmatazing words
    test_data['reviews.text'] = test_data['reviews.text'].apply(lambda x : [WordNetLemmatizer().lemmatize(y) for y in x])

    #removing useless words
    stop = stopwords.words('english')
    test_data['reviews.text'] = test_data['reviews.text'].apply(lambda x: [item for item in x if item not in stop])
    test_data["reviews.text"] = test_data["reviews.text"].apply(lambda x: str(' '.join(x))) #joining all tokens
    sentiment = {1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1}

    test_data["sentiment"] = test_data["reviews.rating"].map(sentiment)
    vectorizer =TfidfVectorizer(max_df=0.9)
    text = vectorizer.fit_transform(test_data["reviews.text"])
    
    converted_vector = trained_tfidf_vector.transform(test_data["reviews.text"])  
    
    return converted_vector


def make_prediction (converted_vector_for_model,test_data,trained_model_rf) :
    
    predicted_result = trained_model_rf[0].predict(converted_vector_for_model)
    predicted_proba = trained_model_rf[0].predict_proba(converted_vector_for_model)
    result_df = test_data[['reviews.text']]
    result_df['actual_rating'] = test_data['reviews.rating']
    result_df['Prediction'] = predicted_result
    result_df['Probability']=np.round(pd.DataFrame(predicted_proba)[1],2)
    sentiment = {0: 'Not satisfied',
            1: "satisfied"}

    result_df["sentiment"] = result_df["Prediction"].map(sentiment)
    
    return result_df


def write_to_s3(bucket_name,result_key, raw_data): # ,aws_access_id,aws_access_key
    csv_buffer = StringIO()
    raw_data.to_csv(csv_buffer)
    resource = boto3.resource('s3') # , aws_access_key_id= aws_access_id ,aws_secret_access_key=aws_access_key
    
    return resource.Object(bucket_name,result_key).put(Body=csv_buffer.getvalue())


def copy_and_delete_prediction_data():
    try:
        #backup the prediction data
        s3 = boto3.resource('s3')
        copy_source = {
                       'Bucket' : bucket_name,
                     'Key'      : inference_data_key
                       }
        s3.meta.client.copy(copy_source,bucket_name,inf_backup_key)
        # deleting the object
        s3_client=boto3.client('s3')
        response =s3_client.delete_object(
                                     Bucket=bucket_name,
                                     Key=inference_data_key
                                            )
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print("File not found : ",  e)
        
        
#s3_conn_id = var_dict["s3_conn_id"]
result_file_location = var_dict["result_file_location"]
result_file_name    =  var_dict["result_file_name"]
pretrained_model_loc = var_dict["pretrained_model_loc"]
bucket_name = var_dict["s3_bucket_name"]
inference_data_folder = var_dict["inference_data_folder"]
inference_file_name = var_dict["inference_file_name"]
backup_path_inf_file = var_dict["inference_file_backup_path"]
backup_filename_inf = var_dict["inference_file_backup_key"]
bucket_name=var_dict['s3_bucket_name']  

#aws_access_id= var_dict['aws_id']
#aws_access_key= var_dict['aws_secret_key']

inference_data_key = inference_data_folder + "/" + inference_file_name
model_key = pretrained_model_loc
result_key = result_file_location +"/" + result_file_name
inf_backup_key = backup_path_inf_file + "/"+ backup_filename_inf

trained_tfidf_vector = pd.read_pickle(r'trained_tfidf_vector.pkl')



def result_func ():
    
    Raw_data = read_csv_file(bucket_name,inference_data_key)
    
    trained_model_rf = load_pickle_data(bucket_name, model_key)
    
    converted_vector = data_cleansing(Raw_data,trained_tfidf_vector)
    
    result_data = make_prediction (converted_vector,Raw_data,trained_model_rf)
    #result_df = result_data.head(10)
    #print(result_df)
    
    write_to_s3(bucket_name,result_key, result_data)
    
    copy_and_delete_prediction_data()
    print("****************Predcition Done successfully and waiting for the next file")
    return result_data




def prediction():
    
    try:
        s3 = boto3.resource('s3')
        s3.Object(bucket_name,inference_data_key).load()
        #print(time.time())

    except ClientError as e:
        print("*************Waiting for the next file to predict*********************")

    else:
        # The object does exist.
        result_func ()

if __name__ == "__main__" :
    
    while True:
        time.sleep(5)
        prediction()


# In[ ]:




