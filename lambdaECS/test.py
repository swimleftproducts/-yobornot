import base64
import requests

def upload_file():
    # Read the file content
    with open('./yob_clip1.wav', 'rb') as f:
        file_content = f.read()

    # Encode the file content as base64
    file_content_base64 = base64.b64encode(file_content).decode('utf-8')

    # Set the request headers
    headers = {
        'Content-Type': 'audio/wave'
    }
    print(type(file_content_base64))
    # Set the request payload
    payload = {
        'body': file_content_base64
    }

    # Send the request to the Lambda function URL
    response = requests.post('https://tam6qsnn5faqzr7jmfmgbue2eu0xzixz.lambda-url.us-east-1.on.aws/', headers=headers, json=payload)

    # Print the response
    print(response.text)


upload_file()
