
# Import necessary libraries and functions
import os
import sys
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

from complement_naive_bayes_classifier.script import train_test

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Complement Naive Bayes Classifier")

    # Add a command-line argument for the file path
    parser.add_argument('file_path', type=str, help='Path to the file to analyze with Complement Naive Bayes Classifier')

    # Parse the command-line arguments
    args = parser.parse_args()
    

    # You can call your run_model function here with the data object
    # Make sure you've imported run_model from ComplementNaiveBayesClassifier

    train_test(args.file_path)

if __name__ == '__main__':
    main()


