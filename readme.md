# Predicting lung cancer survival time

## Prerequisite

You will need following dependencies on your system:

* [Git](http://git-scm.com/)
* [python](https://www.python.org/downloads/)

## Installation

__Python__

Open a linux terminal and run the following commands:

```sh
  $ git clone <repository-url>
  $ cd owkin-challenge && virtualenv -p python3 env
  $ source env/bin/activate
  $ pip install -r requirements.txt
  $ jupyter lab
```

__Docker__

If Docker is available on your computer, you can run the following commands on your terminal:

```sh
  $ git clone <repository-url> && cd owkin-challenge
  $ docker build -t lung-prediction .
  $ docker run -d -e "PORT=4242" -e "PASSWORD=owkin" -p 4242:4242 --name jupyter lung-prediction
```
Now a jupyter server is running on your desktop, you can click on the following link to open the jupyter webui: [jupyter](http://localhost:4242)

The password is "owkin".

