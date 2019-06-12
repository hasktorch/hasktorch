#!/bin/bash

unset LD_LIBRARY_PATH

if [[ ! -e "venv" ]]; then
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

python elmanCell.py
