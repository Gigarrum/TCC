#!/usr/bin/env bash

python train.py 0.0 weights_drop00_balancedBD.pth epochs_data_drop00_balancedBD.csv

python train.py 0.3 weights_drop03_balancedBD.pth epochs_data_drop03_balancedBD.csv

python train.py 0.5 weights_drop05_balancedBD.pth epochs_data_drop05_balancedBD.csv

python train.py 0.8 weights_drop08_balancedBD.pth epochs_data_drop08_balancedBD.csv

