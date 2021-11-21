#!/bin/bash

python3 PrimalSVM.py bank-note/train.csv bank-note/test.csv
python3 DualSVM.py bank-note/train.csv bank-note/test.csv
python3 KernelPerceptron.py bank-note/train.csv bank-note/test.csv
