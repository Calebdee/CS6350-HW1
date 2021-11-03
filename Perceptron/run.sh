!#/bin/bash
echo "Standard Perceptron - 10 Epoches - 0.1 Learning Rate"
python3 perceptron.py bank-note/train.csv bank-note/test.csv

echo ""
echo "======================================================"
echo""

echo "Voted Perceptron - 10 Epoches - 0.1 Learning Rate"
python3 voted-perceptron.py bank-note/train.csv bank-note/test.csv

echo ""
echo "======================================================"
echo ""

echo "Average Perceptron - 10 Epoches - 0.1 Learning Rate"
python3 average-perceptron.py bank-note/train.csv bank-note/test.csv
