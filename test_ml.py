import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, inference, compute_model_metrics

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Verify that the train_model function produces a fitted RandomForestClassifier.
    """

    # Create four simulated observations with one numerical feature each
    X = np.array([[1], [2], [3], [4]])

    # Create two target classes as a controlled binary classification dataset
    y = np.array([0, 0, 1, 1])

    # Train the classifier using the train_model function
    model = train_model(X, y)

    # Confirm the function returns a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)

    # Confirm the model is fitted by checking the presence of the 'classes_' attribute
    assert hasattr(model, 'classes_')

    # Confirm that the classifier recognizes both target classes
    assert set(model.classes_) == {0, 1}

# TODO: implement the second test. Change the function name and input as needed
def test_inference():
    """
    # Verify that the inference functions returns correctly formmated binary predictions.
    """
    # Create four simulated observations with one numerical feature each
    X = np.array([[1], [2], [3], [4]])
    
    # Create two target classes as a controlled binary classification dataset
    y = np.array([0, 0, 1, 1])

    # Train a model so inference can be tested
    model = train_model(X, y)

    # Generate predictions using the four observation
    predictions = inference(model, X)

    # Verify the predictions are returned as a NumPy array
    assert isinstance(predictions, np.ndarray)

    #Verify each observation produced exactly one prediciton
    assert predictions.shape[0] == X.shape[0]

    # Verify the predictions are one of the two target classes (0, 1)
    assert set(predictions).issubset({0, 1})


# TODO: implement the third test. Change the function name and input as needed
def test_salary_column():
    """
    Verify that the salary column exists.
    """

    # Load the census dataset
    data = pd.read_csv("data/census.csv")

    # Verify that the salary column exists in the dataset
    assert "salary" in data.columns
