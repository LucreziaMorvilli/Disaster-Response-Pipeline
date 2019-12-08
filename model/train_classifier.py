import sys
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger'])
nltk.download('stopwords')
import re
import time
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    '''
    Load data on messages and categories from database and return the data in 
    array format for ML model

    Input:
    - database_filepath: where database is stored
    Output:
    - X: array of messages
    - Y: array of categories (labels)
    - target_names: names of categories
    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    target_names = df.iloc[:,4:].columns.tolist()
    
    return X,Y,target_names


def tokenize(text):
    '''
    Tokenize a string by removing stop words, urls, punctuation, upper case and blank spaces 
    and deriving the root of the words by using lemmatizer

    Input:
    - text: string
    Output:
    - clean_tokens: list of strings
    '''
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]",  " ",text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    #stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build a machine learning pipeline with a CountVectorizer, TfidfTransformer 
    and AdaBoostClassifier returning multiple labels. Run GridSearchCV to tune 
    the hyperparameters of the model

    No Input
    Output:
    - pipeline: ML pipeline that can be used to train a model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(),n_jobs=-1))
    ])

    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)), tested already and default are the best
        #'vect__max_df': (0.5, 0.75, 1.0), tested already and default are the best
        'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50,100,200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,verbose = 5, n_jobs=1, 
        scoring='f1_samples')

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model by predicting labels for X_test and returning precision, 
    recall and F-score per category and overall

    Input:
    - model: model to use for predicting labels
    - X_test: messages for which we want to predict categories 
    - Y_test: the true labels for those messages (X_test)
    - category_names: list of the names of all the categories
    Output: print statement
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Save the model using pickle

    Input:
    - model: model you want to save
    - model_filepath: where you want to save the model
    No Output
    '''
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
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
