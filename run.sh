#!/bin/bash

sudo apt-get install python-tk
sudo apt install python3.12-venv

python3 -m venv sfc
. sfc/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py