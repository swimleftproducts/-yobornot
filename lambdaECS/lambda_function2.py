import torch
import boto3
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader





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
model.load_state_dict(torch.load('./model-20230306-014847.pth'))



def predict(model, file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None, duration=3)

    # Downsample to 22050 if necessary
    if sr != 22050:
        audio = librosa.resample(audio, sr, 22050)
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

def test():
    # Predict the label of a file
    file_path = './yob_clip1.wav'
    label = predict(model, file_path)

    # Print the predicted label
    if label == 0:
        print('Not yob')
    else:
        print('Yob')