import sys
import pandas as pd

def main():
	args = sys.argv[1:]
	if len(args) != 2:
		print("well thats just not ok")

	train = pd.read_csv(args[0])
	train_x = train.iloc[:, :6]
	train_y = train.iloc[:, 6:]
	print(train_y)

if __name__ == "__main__":
    main()

