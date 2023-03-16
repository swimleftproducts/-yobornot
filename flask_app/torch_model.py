import boto3
import os
import torch
import torch.nn as nn
import librosa

def download_model(aws_access_key_id, aws_secret_access_key):
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

        bucket_name = 'yobornot'
        bucket_objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='models/')['Contents']
        sorted_objects = sorted(bucket_objects, key=lambda x: x['LastModified'], reverse=True)
        latest_object = sorted_objects[0]
        model_path = latest_object['Key']
        s3_client.download_file(bucket_name, model_path, 'model.pth')
        print('success')
    except:
        print('s3 model grab fialed')

def model():
    model = torch.jit.load('./model.pth')
    model.eval()
    return model

def predict(model, file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None, duration=3)
    
    # Downsample to 22050 if necessary
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        sr = 22050
        
    # Convert audio to mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    
    # Convert to PyTorch tensor and normalize
    mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0)
    mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()
    
    # Move tensor to the same device as the model
    device = next(model.parameters()).device
    mel_spectrogram = mel_spectrogram.to(device)
    
    # Feed the mel spectrogram to the model and get the predicted label
    with torch.no_grad():
        model.eval()
        output = model(mel_spectrogram)
        pred = torch.argmax(output, dim=1).item()
        
    # Return the predicted label
    return pred