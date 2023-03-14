import librosa
from flask import Flask, jsonify

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
