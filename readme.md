This is a flask app that runs as a task in ecs

Running locally:
    set MODE=local
    AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID need to be set as env variables
    run with python app.py


Running locally in docker:
    docker run --rm -p 5000:5000 -e MODE=local flask_app

Running on aws:

    build container 
        docker build -t flask_app .
    log (not always needed)
        aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 216068982475.dkr.ecr.us-east-1.amazonaws.com
    Retag
        docker tag flask_app:latest 216068982475.dkr.ecr.us-east-1.amazonaws.com/yob_or_not:latest

    Push 
        docker push 216068982475.dkr.ecr.us-east-1.amazonaws.com/yob_or_not:latest