


import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from ComplementNaiveBayesClassifier.testing import model_testing

def model_performance(y_test, y_test_pred,best_model_tested):
    
    """
    Evaluate the performance of a machine learning model and generate visualizations.

    Args:
        y_test (numpy.ndarray): True target values for the testing data.
        y_test_pred (numpy.ndarray): Predicted target values for the testing data.
        best_model_tested (ComplementNB): The trained and tested machine learning model.

    Returns:
        Tuple[str, ConfusionMatrixDisplay]:
        - report (str): Classification report including precision, recall, F1-score, and support.
        - confusionmatrix_display (ConfusionMatrixDisplay): Confusion matrix visualization.

    Example:
        To evaluate the model's performance and generate visualizations, use:
        >>> report, confusion_matrix = model_performance(y_test, y_test_pred, best_model_tested)
    """

    #-Model Performance Visualization Confusion of Matrix 
    confusionmatrix_display = ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred,
    labels = best_model_tested.classes_,
    cmap = 'magma'
    )
    #-Classification Report
    report = classification_report(y_test, y_test_pred,zero_division= 0)

    return report,confusionmatrix_display
