FROM python:3.6.10-stretch

WORKDIR /usr/src/app

RUN apt-get -y update && \
    apt-get install --no-install-recommends -y cmake

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY scripts/jupyter_notebook_config.py /root/.jupyter/

COPY scripts/run_jupyter.sh /

RUN chmod +x /run_jupyter.sh

CMD [ "/run_jupyter.sh", "--no-browser", "--allow-root"]
