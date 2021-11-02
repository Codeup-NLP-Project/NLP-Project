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
        
        Arguments:
            - data: Pandas DataFrame
            - classifiers: List of classification models
            - names: Names of classification models
            - lang: Specifies a language to create a lang/not_lang label from
            - top_langs: Specifies the top n langs to create labels for, non-top_langs will be labeled 'other'
    '''
    def __init__(self, data:pd.DataFrame, classifiers: list, names: list, lang = None, top_langs = None):
        ''' Passes dataframe, list of actual classifiers and their names, as well as checks 
            for kwargs lang or top_lang
            Creates a zip of classifiers and their names
        '''
        # Creating class instance of df
        self.df = data.copy(deep = True)
        
        #Checking for individual language specified or n_langs and creating label column
        # For individual lang specification
        if lang != None and top_langs == None: # Checking for lang
            self.lang = lang # assigning lang attribute
            # creating label column
            self.df['label'] = self.df.prog_lang.apply(lambda x: x.lower() if x == self.lang else f'not_{self.lang.lower()}')
        if top_langs != None and lang == None: # Checking for top_langs
            self.top_langs = self.df.prog_lang.value_counts()[:top_langs] # getting top n langs
            # Creating labels column from top n languages            
            self.df['label'] = self.df.prog_lang.apply(lambda x: x.lower() if x in self.top_langs else 'other')
        if lang != None and top_langs != None:
            raise AttributeError('Must specify either lang or top_langs, cant create labels for both.')
        if top_langs != None and top_langs < 2:
            raise AttributeError("Must specify more than one lang, if you want to check for a single language, use lang argument instead.")
        
        # Clean dataframe
        self.df.lemmatized = self.df.lemmatized.apply(basic_clean)
        
        # Creating class attributes
        self.classifiers = classifiers
        self.names = names
        
        models = zip(names, classifiers) # zipping models and names
        self.models = models
        
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
            X_train_validate, X_test, y_train_validate, y_test = train_test_split(self.df.drop(columns = target), 
                                                                            self.df[target],
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
            for word in doc.split(): # iterating through each word in a split doc
                words.append(word) # add to words
        
        word_ser = pd.Series(words) # turn w
        
        # Creating a df from unique words containing raw term count, 
        tf_df = (pd.DataFrame({'raw_count': word_ser.value_counts()})) # raw counts of each term
        tf_df['frequency'] = tf_df.raw_count / tf_df.raw_count.sum() # frequency of each term
        tf_df['augmented_frequency'] = tf_df.frequency / tf_df.frequency.max() # augmented freq of words
        
        return tf_df
    
    def tf_idf(self):
        ''' Gets tf_idf and returns the dataframe of TfidVectorizer
        '''
        tfidf = TfidfVectorizer() # Make the opbject
        bag_of_words = tfidf.fit_transform(self.df['lemmatized'].values) # Fit_transform on lemmatized
        tfidf_df = pd.DataFrame(bag_of_words.todense(), columns=tfidf.get_feature_names()) # Wrapping in a dataframe
        return tfidf_df
    
    def count_vectorize(self, ngram_range = (1,1)):
        ''' Preforms a count vectorizeation with ngrams of n length.
            WARNING: If not cached on system can take a long time to process, 
            creates a cacehd csv for faster use in future iterations.
        '''
        # Checking for cached vectorized csv
        filename = 'Vectorized.csv'
        if os.path.isfile(filename):
            print('Vectorized.csv already exists on your machine! Pulling it now...')
            vector_df = pd.read_csv(filename) # Creating df from csv if present
        else:
            print('''Vectorized.csv does not exist on your local machine! Creating vectorized csv and dataframe now. 
                  Vectorization may take a while, please wait...''')
            # Using Bag of Words count vectorizer for hexamers
            cv = CountVectorizer(ngram_range=(1,1)) # make the object
            vectors = cv.fit_transform(self.df.lemmatized.values) # fit_transform on lemmatized col
            self.vocab_count = cv.vocabulary_
            # Wraps vectorized array in a dataframe with feature names as the columns
            vector_df = pd.DataFrame(vectors.todense(), columns = cv.get_feature_names())
            
            # Create cached csv
            vector_df.to_csv(filename, index=False)
            
            return vector_df
        
        # assigning vectorized dataframe as an attribute
        self.vectorized = vector_df.copy()
        
        return vector_df
        
    
    def metrics(self, metric_type = 'accuracy', splits = 3):
        ''' Checks for and encodes label column
            Creates a metrics df measuring metric_type, accuracy by default.
            Preforms a kfold a number of times determined by splits.
        '''
        try: # checking if label exists, if not raise KeyError, didnt specify a lang or top_langs
            self.df['label']
        except KeyError:
            return KeyError('Must specify language target in class to create models')
        
        try: # Checking if vectorization has already run, if yes there will be an attribute vectorized df
            self.vectorized
        except AttributeError: # If no vectorized attribute exists get vectorized df calling self.count_vectorize
            print('Have not run count_vectorize method yet, running now...')
            self.vectorized = self.count_vectorize()
            print('All done! Moving on to modeling, this may take a while...')
        target = 'label' # Setting target to label
        
        # checking for lang or top_langs
        if self.df[target].nunique == 2: # If one lang chosen
            y_data = self.df[target].replace([f'{self.lang.lower()}', f'not_{self.lang.lower()}'], [1,0]) # Endode lang as 1 not_lang as 0
        else: # if top_langs
            lang_list = [l.lower() for l in list(self.top_langs.index)] # getting a list of all lower case langs in top lang
            lang_list.append('other') # appending 'other' label
            
            lang_encode = list(range(1, len(self.top_langs)+1)) # list of numbers to encode top_langs as
            lang_encode.append(0) # appending 0 for other
            y_data = self.df[target].replace(lang_list, lang_encode) # encoding top_langs
            
            
        result = [] # init empty results list
        for (name, classifier) in self.models: # iterate through zipped models
            kfold = KFold(n_splits = splits) # number of kfolds set to splits
            scores = cross_validate(classifier, self.vectorized, y_data, cv = kfold, scoring = metric_type) # cross validate on each kfold
            result.append(scores) # append to results
            
            msg = "{0}: Accuracy: {1}".format(name, scores['test_score'].mean())
            print(msg)
        
        results = [res['test_score'].mean() for res in result] # list comp to get mean of cross val tests for each model
        metrics_df = pd.DataFrame(data = zip(self.names, results), columns = ['model', metric_type]) # wrap zipped model names and results in dataframe
        return metrics_df.sort_values(by = [metric_type], ascending = False) # return sorted by metric