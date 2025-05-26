#!/bin/bash

chmod +x venv_activator.sh
source venv_activator.sh


T=1000
N=10
D=1
rocThresh=0.7
Tstep=1
nlType=2
lambda=0.01
# Logarithmic grid search values
datasetName="synthetic_RFHORSO"
experimentType="regularizationParameters"


 python3 -u ../syntheticDataExp.py -t $T -lambdaus $lambda  -lambdautr $lambda -n $N -pN "$filename" -D $D -type 1 -threshold $rocThresh -nl $nlType -stdsl 1 -stdsu 1 -exfunctype 1 -de 0.3 -set "$experimentType" -Tstep $Tstep -sslg 1 -ssug 1 -data "$datasetName"