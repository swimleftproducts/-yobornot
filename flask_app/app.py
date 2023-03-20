import librosa
from flask import Flask, jsonify, request
from pathlib import Path
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import os
import torch_model


mode = os.environ['MODE']


if mode == 'local':
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    # Configure AWS credentials
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    torch_model.download_model(aws_access_key_id, aws_secret_access_key)

if mode == 'prod':
    torch_model.download_model(aws_access_key_id=None, aws_secret_access_key=None)

# model = torch_model.model()

# print( torch_model.predict(model,'yob_clip1.wav'))

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/load_audio')
def load_audio():
    audio_file = './yob_clip1.mp3'
    y, sr = librosa.load(audio_file, sr=None)
    length_samples = len(y)
    response = {'length_samples': length_samples, 'sr': sr}
    return jsonify(response)

@app.route('/upload-file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'audio_data' in request.files:
            file = request.files['audio_data']
            # Check if the uploaded file is an MP3 file
            if file.filename.endswith('.mp3'):
                audio_bytes = file.read()
                # Use pydub to convert the MP3 file to WAV format
                from pydub import AudioSegment
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                seg = seg.set_frame_rate(16000)
                seg = seg.set_channels(1)
                wavIO = io.BytesIO()
                seg.export(wavIO, format="wav")
                audio_bytes = wavIO.getvalue()
            elif file.filename.endswith('.wav'):
                audio_bytes = file.read()
            else:
                # Unsupported format, return an error
                return jsonify({'error': 'Unsupported file format'})

            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            length_samples = len(y)
            response = {'length_samples': length_samples, 'sr': sr}
            return jsonify(response)        
    return jsonify('upload failed')





if __name__ == '__main__':
    app.run()
