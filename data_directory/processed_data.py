
#Load the Chirper dataset from a CSV file.
 #Args:
        #dataset_path (str): The file path to the Chirper dataset in CSV format.

import os
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
