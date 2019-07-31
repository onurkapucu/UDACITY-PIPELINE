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
    """ Loads the saved database from the input path and returns target variables, feature variables and category names. """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DF', engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    category_names=list(Y)
    return X, Y, category_names


def tokenize(text):
    """ Tokenizes and lemmatizes the input text
    returns lemmatized clean tokens"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Builds a model pipeline consisting of vectorizer, transformer and classifier.
    Returns a model that uses GridSearch on the created Pipeline
    """
    
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
    """ Evaluates the input model based on test feature and targets.
    Inputs are model, target_test, feature_test and category names respectively"""
    Y_Pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(classification_report(Y_test.as_matrix()[:,i], Y_Pred[:,i]))


def save_model(model, model_filepath):
    """Save the input model to the input filepath as a pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Loads database from the input path and then splits it into test and train datasets. 
    Builds a model which consistst of a pipeline which is fed into GridSearch to be optimized.
    Fits train data to the model and then evaluates the optimized model.
    Finally saves the model to the model_filepath input given during function call.
   
   Inputs: database_filepath to read the saved sqlite database and model_filepath to save the optimized model
    """
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