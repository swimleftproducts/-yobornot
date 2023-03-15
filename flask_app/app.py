import librosa
from flask import Flask, jsonify
import os
import torch_model


mode = os.environ['MODE']

if mode == 'local':
    print(mode)
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    # Configure AWS credentials
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

torch_model.download_model(aws_access_key_id, aws_secret_access_key)
model = torch_model.model()

print( torch_model.predict(model,'yob_clip1.wav'))

app = Flask(__name__)

@app.route('/load_audio')
def load_audio():
    audio_file = './yob_clip1.wav'
    y, sr = librosa.load(audio_file, sr=None)
    length_samples = len(y)
    response = {'length_samples': length_samples, 'sr': sr}
    return jsonify(response)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()
