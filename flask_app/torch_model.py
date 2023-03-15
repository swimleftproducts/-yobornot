import boto3
import os
import torch
import torch.nn as nn
import librosa

def download_model(aws_access_key_id, aws_secret_access_key):
    # Initialize S3 client
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    bucket_name = 'yobornot'
    bucket_objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='models/')['Contents']
    sorted_objects = sorted(bucket_objects, key=lambda x: x['LastModified'], reverse=True)
    latest_object = sorted_objects[0]
    model_path = latest_object['Key']
    model_filename = model_path.split('/')[-1]
    s3_client.download_file(bucket_name, model_path, 'model_weights.pth')
    print('success')

def model():
    class ANN(nn.Module):
        def __init__(self):
            super(ANN, self).__init__()
            self.fc1 = nn.Linear(40*130, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 64)
            self.fc4 = nn.Linear(64, 2)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = x.view(-1, 40*130) # flatten input
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.softmax(self.fc4(x))
            return x
    model = ANN()
    model.load_state_dict(torch.load('./model_weights.pth'))
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