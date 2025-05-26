#!/bin/bash

# Check if the ../venv directory exists
if [ ! -d "../venv" ]; then
  echo "venv directory not found. Creating virtual environment..."
  python3.13 -m venv ../venv
else
  echo "venv directory already exists."
fi


source ../venv/bin/activate