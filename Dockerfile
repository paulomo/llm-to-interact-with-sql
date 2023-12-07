FROM rocker/verse

ENV DEBIAN_FRONTEND=noninteractive

#get python
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev python3.9 python3-pip python3-setuptools python3-dev
RUN pip3 install --upgrade pip
RUN apt -y install libpng-dev && apt install -y gnupg2

# Install odbc
RUN sudo su
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN exit
RUN sudo apt-get update && apt install -y apt-utils
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN sudo apt-get remove -y libodbc2 libodbcinst2 odbcinst unixodbc-common
RUN sudo ACCEPT_EULA=Y apt-get install -y msodbcsql17
#End of mandatory layers for Microsoft ODBC Driver 18 for Linux

# RUN apt-get remove -y curl
#Layers for the django app
RUN mkdir /code
WORKDIR /code
ADD . /code/
RUN pip install pip --upgrade
RUN pip install -r requirements.txt

EXPOSE 8002
WORKDIR /code/pdf_doc_chat
CMD [ "flask", "run","--host","0.0.0.0","--port","8002"]