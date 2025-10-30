import pickle
import os
from helpers import create_csv_submission
from preprocessing_subroutine import preprocess_data
from model_training_subroutine import model_training
from model_training_functions import classify_test_data, compute_f1_on_train

MODEL_PATH = "final_logreg_model.pkl"

# Check if the model file exists
if os.path.exists(MODEL_PATH):
    # Load the trained model using pickle
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
else:
    # Train the model (saves to MODEL_PATH)
    model_training()

    # Load the trained model
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)

# Get preprocessed test data (x_test_final) from preprocessing subroutine
x_train_final, x_test_final, y_train, train_ids, test_ids = preprocess_data(
    verbose=True
)

# Print Loss on training data
print(f"Training Loss: {model['loss']}")

# Compute F1 score on training data
compute_f1_on_train(
    model=model, x_train_final=x_train_final, y_train=y_train, verbose=True
)

# Classify test data
y_test = classify_test_data(x_test_final, model)["yhat_label_pm1"]

# Save predictions to CSV file for submission
create_csv_submission(test_ids, y_test, "submission.csv")
