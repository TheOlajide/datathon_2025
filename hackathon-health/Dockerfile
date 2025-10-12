# we got this image version
FROM python:3.9-slim
### now we need to set our working directory as app so that everything should be done inside the app
WORKDIR /app 

## we need to install system packages 
#apt-get update is response for getting new latest packages
##apt-get install for installation
## gcc since we need the compile for our xgboost and some of the librlaies
## libpq-dev its a postgress clien libraly that conncets to the redshift
### rm -rf /var/lib/apt/lists this will clean the package cache to increase the speed

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

### we are just copying this files which has our dependencies for docker to find it
COPY requirements.txt .
### then installing those dependencies 
RUN pip install --no-cache-dir -r requirements.txt

## now copying everythingg rom my local directories
COPY . .
### we are now exposing the port 
EXPOSE 8000
#### default command for the running of this application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]