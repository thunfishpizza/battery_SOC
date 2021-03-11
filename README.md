## Battery Intelligence Case Study
Data science case study for battery data, obtained from RWTH Aachen data repository. The workflow includes problem definition, exploratory analysis and implementation of a predictive model for battery state of charge. This project has been carried out in approximately 2-3 days.

## Environment
The stack is set up on a 64-bit MacOs11 desktop with Conda 4.9.2 and Python 3.7.6. A new environment is created and activated via terminal. To recreate:
1. Clone the project Git repository.
2. Create a conda environment for the accure project from within the project's root directory:
> conda env create --force -f environment.yml -n accure <br/>
> conda activate accure

The "main.py" script executes the whole workflow and saves the output as a logfile in data/output. Total execution time is around 10 mins.
> python main.py

## Overview
* Problem definition and exploratory data analysis.
* Data preprocessing (outliers, scaling) and feature engineering.
* Spot checking various linear and non-linear regression models to choose the best.
* Model fitting, tuning and evaluation of a Random Forest learner on the test dataset.
* Model achieved a train RMSE of 3.0 and test RMSE of 3.7, which is better than the naive baseline of 32 (mean prediction).

## Project Organization
```

├── main.py                             <- Main python file for execution.
│
├── exploration.ipynb                   <- Jupyter Notebook to perform exploratory analysis, feature engineering,
|                                           model selection, parameter tuning and validation.
│
├── model                               <- Trained and serialized RF model.
│
│
├── data
│   ├── raw                             <- The original, immutable data dump.
│   ├── processed                       <- Processed dataset.
│   ├── output                          <- Predicted output with true labels.
│
├── source
│   ├── Requirements.txt                <- Required Python packages.
│   ├── fileRead.py                     <- Input fixtures for pytests.
│   ├── preProcessing.py                <- Module containing pytest function for main.py.
│   ├── modelling.py                    <- Module containing pytest for ranker model.    
│   └── prediction.py                   <- Module containing pytest for remote data retrieving.
│ 
├── environment.yml                     <- Required Python packages (refers to Requirements.txt).
├── Makefile                            <- Makefile to control the project.
├── README.md                           <- The top-level README for users.
├── setup.py                            <- Module with the package install script.
│ 
├── logs
│   ├── logfile<timestamp>.log          <- Log of the executions. 
│ 
├── .gitignore                          <- Files to be ignored by git.
