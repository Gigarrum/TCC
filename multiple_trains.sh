#!/usr/bin/env bash

python train.py 0.3 weights_drop03.pth epochs_data_drop03.csv

python train.py 0.5 weights_drop05.pth epochs_data_drop05.csv

python train.py 0.8 weights_drop08.pth epochs_data_drop08.csv

