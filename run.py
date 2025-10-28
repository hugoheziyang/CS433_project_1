import h5py
import numpy as np
import os
from helpers import create_csv_submission
from preprocessing_subroutine import preprocess_data
from model_training_subroutine import model_training
from model_training_functions import classify_test_data

# Check if the model file exists
if os.path.exists("final_logreg_model.h5"):
    # Load the trained model
    with h5py.File("final_logreg_model.h5", "r") as f:
        model = {key: f[key][()] for key in f.keys()}

else:
    # Train the model
    model_training()
    
    # Load the trained model
    with h5py.File("final_logreg_model.h5", "r") as f:
        model = {key: f[key][()] for key in f.keys()}

# Get preprocessed test data (x_test_final) from preprocessing subroutine
x_train_final, x_test_final, y_train, train_ids, test_ids = preprocess_data()

# Classify test data
y_test = classify_test_data(x_test_final, model)["yhat_label_pm1"]

# Save predictions to CSV file for submission
create_csv_submission(test_ids, y_test, "y_test.csv")

    


