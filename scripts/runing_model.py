
# Import necessary libraries and functions
import os
import sys
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

from ComplementNaiveBayesClassifier.model_functions5 import run_model

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Complement Naive Bayes Classifier")

    # Add a command-line argument for the file path
    parser.add_argument('file_path', type=str, help='Path to the file to analyze with Complement Naive Bayes Classifier')

    # Parse the command-line arguments
    args = parser.parse_args()
    

    # You can call your run_model function here with the data object
    # Make sure you've imported run_model from ComplementNaiveBayesClassifier

    run_model(args.file_path)

if __name__ == '__main__':
    main()


