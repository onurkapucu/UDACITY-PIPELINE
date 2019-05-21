import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
	"""load the cleaned and merged database file from the input path"""
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DF', engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    category_names=list(Y)
    return X, Y, category_names


def tokenize(text):
	"""tokenizes and then lemmatizes the input text using nltk's wordtokenize and WordNetLemmatizer libraries"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
	"""Create a pipeline consisting of CountVectorizer Tfidf transformer and RandomForestClassifier. This function also runs GridSearch to optimize the model parameters"""

    #create the model pipeline with optimized variables
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
         ])


    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 500,1000,5000,7500,1000),
        'tfidf__use_idf': (True, False)
    }
    model  = GridSearchCV(pipeline, param_grid=parameters,refit=True)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
	"""Evaluates model performance for every label for the given test data"""
    Y_Pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(classification_report(Y_test.as_matrix()[:,i], Y_Pred[:,i]))


def save_model(model, model_filepath):
	"""Extracts and saves the model as a pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
	"""This script loads the database and splits the inputs/outputs into test and train. Then creates a model and optimizes it. Prints the model performance for each output category and then saves the model as a pickle file"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()