
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn.naive_bayes import MultinomialNB,ComplementNB



def load_data(data_path):
    """
    Load a dataset from a CSV file.

    Args:
        data_path (str): The file path to the dataset in CSV format.
            The `data_path` should be a valid file path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded dataset.

    Example:
        To load a dataset, use:
        >>> data = load_data('/path/to/your/dataset.csv')
        >>> print(data)
    """
    data = pd.read_csv(data_path, encoding="ISO-8859-1")
    return data


def data_segmentation(data):
    """
    Split the data into training and testing sets for machine learning.

    Args:
        data (pandas.DataFrame): The dataset containing 'Text' and 'Target' columns.

    Returns:
        Tuple[pandas.Series, pandas.Series, pandas.Series, pandas.Series]: A tuple containing the following data splits:
            - x_train: Training input data (text)
            - x_test: Testing input data (text)
            - y_train: Training target data (sentiment labels)
            - y_test: Testing target data (sentiment labels)

    Example:
        To split the dataset into training and testing sets, use:
        >>> x_train, x_test, y_train, y_test = data_segmentation(your_data)
    """

    inputs = data['Text']
    target = data['Target']
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, target, 
                                                        test_size=0.3, 
                                                        random_state=365, 
                                                        stratify=target)
    
    return x_train, x_test, y_train, y_test



def feature_engineering(x_train, x_test, y_train, y_test):
    """
    Perform feature engineering on text data.(Text Data Transformation from Text to Numbers)

    Args:
        x_train (pandas.Series): Training input data containing text.
        x_test (pandas.Series): Testing input data containing text.
        y_train (pandas.Series): Training target data.
        y_test (pandas.Series): Testing target data.

    Returns:
        Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, CountVectorizer]:
        - x_train_transf: Transformed training input data (text to numbers).
        - x_test_transf: Transformed testing input data (text to numbers).
        - vectorizer: CountVectorizer object used for transformation.

    Example:
        To perform text data transformation, use:
        >>> x_train_transf, x_test_transf, vectorizer = feature_engineering(x_train, x_test, y_train, y_test)
    """
    vectorizer = CountVectorizer()
    x_train_transf = vectorizer.fit_transform(x_train)
    x_test_transf = vectorizer.transform(x_test)
    
    return x_train_transf,x_test_transf,vectorizer



def model_training(x_train_transf,y_train):
    
    """
    Train a machine learning model using the training data.

    Args:
        x_train_transf (scipy.sparse.csr_matrix): Transformed training input data.
        y_train (pandas.Series): Training target data.

    Returns:
        ComplementNB: The trained machine learning model.

    Example:
        To train a model, use:
        >>> trained_model = model_training(x_train_transf, y_train)
    """
    ### 5- Model Training
    best_model = ComplementNB()
    best_model.fit(x_train_transf, y_train)
    
    return best_model



def model_testing(best_model,x_test_transf):
    """
    Test a machine learning model on the testing data and evaluate its performance.

    Args:
        best_model (ComplementNB): The trained machine learning model.
        x_test_transf (scipy.sparse.csr_matrix): Transformed testing input data.

    Returns:
        Tuple[ComplementNB, numpy.ndarray, numpy.ndarray]:
        - best_model: The trained machine learning model.
        - y_test_pred: Predicted labels for the testing data.
        - prior_probability: Class prior probabilities.

    Example:
        To test the model and evaluate its performance, use:
        >>> trained_model, y_test_pred, prior_prob = model_testing(best_model, x_test_transf)
    """
    ### 6- Class Prior Probability Transformation.
    #The numbers you're seeing are probabilities, and probabilities are often expressed 
    #as decimals between 0 and 1.
    prior_probability = np.exp(best_model.class_log_prior_)

    ### 7- Model Prediction and Evaluation.
    y_test_pred = best_model.predict(x_test_transf)

    return best_model,y_test_pred,prior_probability



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

