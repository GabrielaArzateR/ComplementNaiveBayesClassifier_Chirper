import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


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