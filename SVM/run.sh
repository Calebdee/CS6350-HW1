#!/bin/bash

python PrimalSVM.py bank-note/train.csv bank-note/test.csv
python DualSVM.py bank-note/train.csv bank-note/test.csv
python KernelPerceptron.py bank-note/train.csv bank-note/test.csv
