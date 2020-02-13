# Predicting lung cancer survival time

## Prerequisite

Vous aurez besoin des programmes suivants installés sur votre ordinateur :

* [Git](http://git-scm.com/)
* [python](https://www.python.org/downloads/) ou [anaconda](https://repo.continuum.io/)

## Installation

__Python__

Open a linux terminal and run the following commands:

```sh
  $ git clone <repository-url>
  $ cd orfee && virtualenv -p python3 env
  $ source env/bin/activate
  $ pip install -r requirements.txt
  $ jupyter notebook
```

__Docker__

If Docker is available on your computer, you can run the following commands on your terminal:

```sh
  $ git clone <repository-url> && cd orfee
  $ docker build -t lung-prediction .
  $ docker run -d -e "PORT=4242" -e "PASSWORD=owkin" -p 4242:4242 --name jupyter lung-prediction
```
Now a jupyter server is running on your desktop, you can click on the following link to open the jupyter webui: [jupyter](http://localhost:4242)

The password is "owkin".

