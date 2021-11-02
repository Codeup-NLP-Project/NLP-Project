import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import warnings
warnings.simplefilter('ignore')


class NLP_model():
    ''' Creates classification models using a variety of Sklearn models.

        Models:
        ----------------------------------------------------------------
        KNeighborsClassifier, DecisionTreeClassifier, svm, GaussianNB, 
        MultinomialNB, GaussianProcessClassifier, MLPClassifier, RandomForestClassifier, AdaBoostClassifier
        ----------------------------------------------------------------
    '''
    def __init__(self, data:pd.DataFrame, classifiers: list, names: list):
        ''' Passes dataframe, list of actual classifiers, and names of the classifiers.
            Creates a zip of classifiers and their names
        '''
        
        df = data.copy(deep = True)
        
        # Clean dataframe
        df.lemmatized = df.lemmatized.apply(basic_clean)
        
        # Creating class attributes
        self.classifiers = classifiers
        self.names = names
        
        # Creating class instance of df
        self.df = df.copy(deep = True)
        
        # Checking equal lengths of names and classifiers
        if len(names) == len(classifiers):
            self.models = zip(names, classifiers) # if equal length make models zip
        else:
            raise ValueError('List of classifiers and names must be the same length')
        
    def split(self, target = None):
        '''
        This function takes in a dataframe and, optionally, a target_var array. Performs a train, validate, 
        test split with no stratification. Returns train, validate, and test dfs.
        '''
        
        # Checking for y specified
        if target is None: # if no y, preform regular train, validate, test split
            train_validate, test = train_test_split(self.df, test_size=.2, 
                                                    random_state=1312)
            train, validate = train_test_split(train_validate, test_size=.3, 
                                                    random_state=1312)
            
            self.train, self.validate, self.test = train, validate, test # setting self versions of each df
            return train, validate, test
        
        # If y is specified preform X/y train, validate, test split
        else:
            X_train_validate, X_test, y_train_validate, y_test = train_test_split(self.df, target,
                                                                            test_size=.2, 
                                                                            random_state=1312)
            X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate, y_train_validate,
                                                                            test_size=.3, 
                                                                            random_state=1312)
            self.X_train, self.X_validate, self.X_test,\
            self.y_train, self.y_validate, self.y_test = X_train, X_validate, X_test, y_train,\
                                                        y_validate, y_test # attributes for each X/y df and array
            
            return X_train, X_validate, X_test, y_train, y_validate, y_test
    
    
    def tf(self):
        ''' Gets the term frequency of lematized column in the df
        '''
        
        # For each lemmatized doc, append to series
        docs = [] # init empty series for split documents
        words = [] # init empty series for unique words
        for doc in self.df['lemmatized'].values:
            docs.append(doc.split()) # append list of split words
            for word in docs: # iterating through each word in a split doc
                words.append(word) # if unique add to words
        
        # Creating a df from unique words containing raw term count, 
        tf_df = (pd.DataFrame({'raw_count': pd.Series(words).value_counts()}) # raw counts of each term
        .assign(frequency=lambda x: x.raw_count / x.raw_count.sum()) # frequency of each term
        .assign(augmented_frequency=lambda x: x.frequency / x.frequency.max())) # augmented freq of words
        
        return tf_df
    
    #TODO Create tf_idf method
    # def tf_idf(self):
    #     return
    
        
    def metrics(self, metric_type = 'accuracy', splits = 10):
        ''' Creates a metrics df measuring metric_type. 
        
        '''
        results = []
        for i, (name, classifier) in enumerate(models):
            kfold = KFold(n_splits = splits)
            scores = cross_validate(model, X_data, y_data, cv = kfold, scoring = 'accuracy', return_estimator=True)
            result.append(scores)

            #TODO Finish creating metrics method to return metrics after model

def classifier_models(X_data, y_data, classifier_names, classifier_models):
    '''
        Takes two arrays:
        - X_data = data without the target_var included
        - y_data = an array of the target_var
        - List of model names 
        - List of the classifiers themselves
        
        Preforms K-fold and cross-validation and returns a metrics dataframe with the model name and accuracy score. 
    '''
    # Zipping models and Classifiers
    models = zip(classifier_names, classifier_models)

    # Init empty lists
    names = [] 
    result = []
    coeff = []

    # Cross-validating accuracy for each model based on Train subset
    for i, (name, model) in enumerate(models):
        kfold = KFold(n_splits = 10)
        scores = cross_validate(model, X_data, y_data, cv = kfold, scoring = 'accuracy', return_estimator=True)
        result.append(scores)
        names.append(name)
        try:
            coeff.append(model.coeff_)
        except AttributeError:
            coeff.append(None)
        msg = "{0}: Accuracy: {1}, Coeff: {2}".format(name, scores['test_score'].mean(), coeff[i])
        print(msg)
        
    results = [res['test_score'].mean() for res in result]
    metrics_df = pd.DataFrame(data = zip(names, results), columns = ['Model', 'Accuracy'])
    return metrics_df.sort_values(by = ['Accuracy'], ascending = False)