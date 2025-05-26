#!/bin/bash

chmod +x venv_activator.sh
source venv_activator.sh

lambda=0.0

python3 ./realDataExp.py -data noaa_coastwatch_cellular -lambdaus $lambda -lambdautr $lambda -D 1 -P 4 -Tstep 1