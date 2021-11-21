!#/bin/bash
echo "Standard Perceptron - 10 Epoches - 0.1 Learning Rate"
python perceptron.py bank-note/train.csv bank-note/test.csv

echo ""
echo "======================================================"
echo""

echo "Voted Perceptron - 10 Epoches - 0.1 Learning Rate"
python voted-perceptron.py bank-note/train.csv bank-note/test.csv

echo ""
echo "======================================================"
echo ""

echo "Average Perceptron - 10 Epoches - 0.1 Learning Rate"
python average-perceptron.py bank-note/train.csv bank-note/test.csv
