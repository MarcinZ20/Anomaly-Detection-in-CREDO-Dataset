import os
from pathlib import Path
import csv

def annotateDataset() -> None:
    for file in os.listdir(r'../../data/processed'):
        # create a csv file that contains the names of all the files in the processed folder
        with open(r'../../data/processed_files.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([file])
        