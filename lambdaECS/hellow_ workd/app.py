import sys
import librosa
import base64
import io

def handler(event, context):
    response_headers = {
        'Content-Type': 'audio/wav',
        'Access-Control-Allow-Origin': '*'
    }
    print('in lambda')
    # Check if the request contains a file
    if 'body' in event and event['body'] is not None:
        # Get the file content
        file_content = event['body']

        # Decode the base64-encoded content
        file_content_decoded = base64.b64decode(file_content)

        # Load the audio file from the binary content
        with io.BytesIO(file_content_decoded) as f:
            audio, sr = librosa.load(f, sr=None, duration=3)

        # Return a success response
        return {
             'headers': response_headers,
            'statusCode': 200,
            'body': f'File received with {sr} and saved successfully'
        }
    try:
        return 'Hello from AWS Lambda using Python {} and librosa is installed!'.format(sys.version)
    except ImportError:
        return 'Hello from AWS Lambda using Python {} and librosa is NOT installed!'.format(sys.version)
