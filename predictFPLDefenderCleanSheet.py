import argparse
import pandas as pd
import pickle
from helpers import predict_clean_sheet, print_predictions

# Create an argument parser object
parser = argparse.ArgumentParser(description='Parse gameweek')

# Add custom arguments
parser.add_argument('-g', '--gw', type=str, help='Gameweek')

# Parse the arguments provided by the user
args = parser.parse_args()

# Use the arguments in your script
gw = 0
if args.gw:
    gw = args.gw

if gw <= '0' or gw > '38':
    print ("Provide a valid Game week value")
else:
    gw_fixtures, predictions = predict_clean_sheet(gw)
    print_predictions(gw_fixtures, predictions)

