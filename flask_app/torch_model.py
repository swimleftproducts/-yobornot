import boto3
import os

def download_model():
    # Initialize S3 client
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    bucket_name = 'yobornot'
    bucket_objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='models/')['Contents']
    sorted_objects = sorted(bucket_objects, key=lambda x: x['LastModified'], reverse=True)
    latest_object = sorted_objects[0]
    model_path = latest_object['Key']
    model_filename = model_path.split('/')[-1]
    s3_client.download_file(bucket_name, model_path, model_filename)