### Coronary Heart Disease Prediction (EPFL Fall 2025)

---

## Project overview
This project implements a **machine learning pipeline** to predict the likelihood of **coronary heart disease (CHD)** using health-related data from the **Behavioral Risk Factor Surveillance System (BRFSS)** dataset. For a detailed description of the dataset used, see [EPFL Machine Learning Project 1 – AIcrowd Challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)

---

## Authors
- Ziyang He  
- Romeo Estezet  
- Capucine Denis  

---

## Requirements
Numpy and Matplotlib external libraries are required to run code. To install them, run in command line:

```bash
python -m venv .venv
source .venv/bin/activate     
pip install -r requirements.txt
```

--- 

## Dataset repository
The files:
- `x_train.csv`
- `y_train.csv`
- `x_test.csv` 

must be located in `./data/dataset` repository. This can be done simply by clicking on `dataset.zip` in the `./data` repository, which should automatically create `./data/dataset` repository.

---

## Running model to create `submission.csv` file
To create `.csv` submission on [EPFL Machine Learning Project 1 – AIcrowd Challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1), run the following in command line:

```bash
source .venv/bin/activate
python run.py
```

This should produce a file named `submission.csv`, which can then be uploaded onto the competition webpage. 

---

## Files description
The objectives of the assignment is described in `project1_description.pdf`. We describe the files used in the execution of `run.py` and/or the written report below:
- `helpers.py` contains functions which load `x_train.csv`, `y_train.csv`, `x_test.csv` data into numpy arrays as well as create `.csv` submission file.
- `implementations.py` contains functions required in **Step 2 - Implement ML Methods** of `project1_description.pdf` assignment as well as auxiliary functions needed.
- `preprocessing_subroutine.py` contains the data preprocessing subroutine. 
- `preprocessing_functions.py` contains auxiliary functions to the data preprocessing subroutine in `preprocessing_subroutine.py`.
- `model_training_subroutine.py` contains the subroutine used to train a given model. The model may either be regularised logistic regression, or ridge regression. 
- `model_training_functions.py` contains the auxiliary functions to the model training subroutine in `model_training_subroutine.py`.
- `plot_graphs.py` contains functions used to plot the graphs seen in the written report `project1_report.pdf`.

---

## Cached data
To simplify loading data from `.csv` files and converting them into numpy arrays each time we manipulate the training dataset, we load from `x_train.csv`, `y_train.csv`, `x_test.csv` once, and store the numpy arrays in `./data/dataset/cached_final_data.npz`. This is done the first time `preprocess_data` in `preprocessing_subroutine.py` is executed, and upon future calls of `preprocess_data`, automatically loads dataset from `./data/dataset/cached_final_data.npz`.

The trained logistic regression model is stored in `final_logreg_model.pkl` upon running `model_training` subroutine in `model_training_subroutine.py`. This is done automatically upon executing `run.py`. 

---

## Other repositories
- `./tests` includes tests used to verify certain preprocessing functions
- `./figures` includes the graphs used in the written report `project1_report.pdf`