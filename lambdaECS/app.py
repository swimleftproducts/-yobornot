import base64
import librosa
import io

def lambda_handler(event, context):
    print('in lambda')
    # Check if the request contains a file
    if 'body' in event and event['body'] is not None:
        # Get the file content
        file_content = event['body']
        
        # Load the audio file from the binary content
        with io.BytesIO(file_content) as f:
            audio, sr = librosa.load(f, sr=None, duration=3)
        
        # Return a success response
        return {
            'statusCode': 200,
            'body': f'File received with {sr} and saved successfully'
        }
    else:
        # Return an error response
        return {
            'statusCode': 400,
            'body': 'File not found in request'
        }