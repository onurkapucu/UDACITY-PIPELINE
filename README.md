# UDACITY-PIPELINE

This project aims to create a pipeline to classify text inputs into 36 categories to better respond disaster/critical situation messages.

## Content

### 1- Notebooks 
This folder covers the work to develop the code for data cleaning/preperation and ML Pipeline building and optimization. 

ETL Pipeline Preperation notebook cleans the given training data and creates a database

ML Pipeline Preperation Notebook loads the database and develops the pipeline to estimate significance of the message within the predefined 36 categories

### 2- Data

You can find the data and a standalone python code, process_data.py, derived from ETL Pipeline Preperatiob Notebook.

A sample code to run process_data.py:

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`


### 3- models

You can find the standalone python code, train_classifier.py, derived from ML Pipeline Preparation Notebook. This code will read the database created by proces_data.py. It will split the data into test and train and create a model then export it into a pickle file of specified name.

A sample code to run train_classifier.py:

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

#### 4- app

This folder contains a web app that can be run using run.py. By default it will display the distribution of database message genres. Also allows to test and classify random inputs.
