import numpy as np

def model_testing(best_model, x_test_transf):
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