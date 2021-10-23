import sys
import pandas as pd
import math

def main():
	args = sys.argv[1:]
	if len(args) != 2:
		print("well thats just not ok")
	train = pd.read_csv(args[0], header=None)
	test = pd.read_csv(args[1], header=None)


if __name__ == "__main__":
    main()