# UDACITY-PIPELINE

This project aims to create a pipeline to classify text inputs into different categories. Data provided in the project is using tweets categorize them into 36 disaster/critical situation categories. 

Target is to be able to differentiate critical situation/disaster related texts from non critical ones.

## File Content

### 1- Notebooks 
This folder covers the work to develop the code for data cleaning/preperation and ML Pipeline building and optimization. 

ETL Pipeline Preperation notebook cleans the given training data and creates a database

ML Pipeline Preperation Notebook loads the database and develops the pipeline to estimate significance of the message within the predefined 36 categories

### 2- Data

You can find the data and a standalone python code, process_data.py, derived from ETL Pipeline Preperatiob Notebook.

A sample code to run process_data.py:

	python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db


### 3- Models

You can find the standalone python code, train_classifier.py, derived from ML Pipeline Preparation Notebook. This code will read the database created by proces_data.py. It will split the data into test and train and create a model then export it into a pickle file of specified name.

A sample code to run train_classifier.py:

	python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

#### Libraries:

**nltk:** Natural language processing toolbox to tokenize, lemmatize texts

**numpy:** Numerical operations

**pandas:** handle data easily in DataFrame

*sqlalchemy:* to create and interact with sql databases

**sklearn.pipeline:** Used Pipeline to easily handle/modify and optimize dataprocessing in sequence

**sklearn.multioutput:** Used MultiOutputClassifier to be able to handle classification when there are multiple outputs(36 in given dataset)

**sklearn.feature_extraction.text:** CountVectorizer and TfidfTransformer for text vectorization and transformation

**sklearn.ensemble:** Used RandomForestClassifier, AdaBoostClassifier as classifiers

**sklearn.model_selection:** Used train_test_split to split data into test and train datasets

**sklearn.model_selection:** Used GridSearchCV for pipeline parameter optimization

**pickle:**: To save and reuse models

#### Model Pipeline: 

I used a model pipeline that consists of a vectorizer, transformer and classifier:

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
         ])

This pipeline is then optimized using GridSearch for following parameter values:

	parameters = {
		'vect__ngram_range': ((1, 1), (1, 2)),
		'vect__max_df': (0.5, 0.75, 1.0),
		'vect__max_features': (None, 500,1000,5000,7500,1000),
		'tfidf__use_idf': (True, False) }



#### 4- App

This folder contains a web app that can be run using run.py. By default it will display the distribution of database message genres. Also allows to test and classify random inputs.
